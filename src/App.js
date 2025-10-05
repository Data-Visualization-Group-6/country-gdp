import React, { useState, useEffect, useMemo, useRef } from "react";
import { Upload } from "lucide-react";
import * as d3 from "d3";
import { voronoiTreemap } from "d3-voronoi-treemap";
import Papa from "papaparse";

const TOP_N_PER_CONTINENT = 9;

const continentColors = {
  Africa: "#FF6B6B",
  Asia: "#008000",
  Europe: "#45B7D1",
  "North America": "#FFA07A",
  "South America": "#98D8C8",
  Oceania: "#F7DC6F",
};

const gdpComponentColors = {
  "Agriculture (% GDP)": "#228B22",
  "Industry (% GDP)": "#4169E1",
  "Service (% GDP)": "#FFD700",
  "Export (% GDP)": "#8B008B",
  "Import (% GDP)": "#FF4500",
  Other: "#E0E0E0",
};

function getInflationColor(inflationRate) {
  if (inflationRate === null || inflationRate === undefined || isNaN(inflationRate)) return "#FFFFFF";
  if (inflationRate < 0) {
    const intensity = Math.min(Math.abs(inflationRate) * 20, 255);
    return `rgb(255, ${255 - intensity}, ${255 - intensity})`; // red-ish for deflation magnitude
  } else {
    const intensity = Math.min(inflationRate * 20, 255);
    return `rgb(${255 - intensity}, 255, ${255 - intensity})`; // green-ish for inflation magnitude
  }
}

function getOpacity(unemployment) {
  if (unemployment === null || unemployment === undefined || isNaN(unemployment)) return 0.5;
  const employment = 100 - unemployment;
  return Math.max(0.3, Math.min(1, employment / 100));
}

function calcMakeup(record) {
  const comp = {
    "Agriculture (% GDP)": Number(record["Agriculture (% GDP)"]) || 0,
    "Industry (% GDP)": Number(record["Industry (% GDP)"]) || 0,
    "Service (% GDP)": Number(record["Service (% GDP)"]) || 0,
    "Export (% GDP)": Number(record["Export (% GDP)"]) || 0,
    "Import (% GDP)": Number(record["Import (% GDP)"]) || 0,
  };
  const total = Object.values(comp).reduce((s, v) => s + v, 0);
  const other = Math.max(0, 100 - total);
  return { ...comp, Other: other };
}
function numberConversion(number){
  if (number >= 1000000 && number <= 999999999){
    return (number/1000000).toFixed(2) + "M";
  }else if (number >= 1000000000 && number <= 999999999999){
    return (number/1000000000).toFixed(1) + "B";
  }else if (number >= 1000000000000 && number <= 999999999999999){
    return (number/1000000000000).toFixed(2) + "T";
  
} 
}

function buildHierarchy(rows, year) {
  const yearRows = rows.filter(
    (r) => Number(r.Year) === Number(year) && r["Country Name"] && Number(r.GDP) > 0
  );
  

  // limit to TOP_N per continent + Others (adjust TOP_N_PER_CONTINENT to taste)
  const byCont = d3.group(yearRows, (d) => d["Continent Name"] || "Unknown");
  const picked = [];

  for (const [cont, items] of byCont.entries()) {
    const sorted = items.slice().sort((a, b) => Number(b.GDP) - Number(a.GDP));
    const top = sorted.slice(0, TOP_N_PER_CONTINENT);
    const rest = sorted.slice(TOP_N_PER_CONTINENT);
    picked.push(...top);
    if (rest.length) {
      const othersGDP = d3.sum(rest, (d) => Number(d.GDP) || 0);
      const avgUnemp = d3.mean(rest, (d) => Number(d.Unemployment) || 0);
      const avgInfl = d3.mean(rest, (d) => Number(d["Inflation Rate"]) || 0);
      picked.push({
        Year: year,
        "Country Name": `Others (${cont})`,
        "Continent Name": cont,
        GDP: othersGDP,
        Unemployment: avgUnemp,
        "Inflation Rate": avgInfl,
        // no makeup for aggregate; render as a solid cell
      });
    }
  }

  const children = picked.map((r) => {
    const country = {
      name: r["Country Name"],
      continent: r["Continent Name"] || "Unknown",
      unemployment: Number(r.Unemployment).toFixed(0),
      inflation: Number(r["Inflation Rate"]).toFixed(1),
      service: Number(r["Service (% GDP)"]).toFixed(1),
      import: Number(r["Import (% GDP)"]).toFixed(1),
      export: Number(r["Export (% GDP)"]).toFixed(1),
      agriculture: Number(r["Agriculture (% GDP)"]).toFixed(1),
      industry: Number(r["Industry (% GDP)"]).toFixed(1),
      gpdpercapita: Number(r["GDP Per Capita"]).toFixed(2),
      education:Number(r["Education Expenditure"]).toFixed(0),
      health: Number(r["Health Expenditure"]).toFixed(0)
      // If we have makeup fields, add children; else make leaf with value = GDP
    };

    const makeup = calcMakeup(r);
    const entries = Object.entries(makeup).filter(([, pct]) => pct > 0.0001);
    if (entries.length > 0 && !country.name.startsWith("Others (")) {
      return {
        ...country,
        // no direct value; sum of components defines size == GDP
        children: entries.map(([label, pct]) => ({
          name: label,
          value: (Number(r.GDP) || 0) * (pct / 100),
        })),
      };
    } else {
      // Leaf with no subdivisions
      return {
        ...country,
        value: Number(r.GDP) || 0,
      };
    }
  });

  return { name: "World", children };
}

const VoronoiTreemap = () => {
  const [rows, setRows] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [yearBounds, setYearBounds] = useState([2000, 2022]);
  const [selectedYear, setSelectedYear] = useState(2000);
  const [displayMode, setDisplayMode] = useState("name"); // 'name' or 'makeup'
  //const [toolTip,setToolTip] = useState(null) //state used to keep track of tooltip instance

  const wrapperRef = useRef(null);
  const svgRef = useRef(null);
  const [dims, setDims] = useState({ w: 1000, h: 700 });
  const toolTipRef = useRef(null); // ref for toolTip object

  // Resize observer for responsive SVG
  useEffect(() => {
    const ro = new ResizeObserver((entries) => {
      const w = Math.max(640, Math.floor(entries[0].contentRect.width));
      setDims({ w, h: Math.max(500, Math.round(w * 0.62)) });
    });

    //tooltip creation when page is first rendered
    toolTipRef.current = d3.select("body")
      .append("div")
      .attr("class","tooltip")
      .style("position", "absolute")
      .style("opacity",0)
      .style("pointer-events","none")
      .style("background","rgba(0,0,0,0.8)")
      .style("color", "#fff")
      .style("padding", "6px 10px")
      .style("border-radius", "4px")
      .style("font-size", "12px"); 
      

    if (wrapperRef.current) ro.observe(wrapperRef.current);
    return () => ro.disconnect();


  }, []);

  // Parse CSV via Papa
  const onFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setProcessing(true);
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: ({ data }) => {
        const coerced = data.map((r) => ({
          ...r,
          Year: Number(r.Year),
          GDP: Number(r.GDP),
          Unemployment: r.Unemployment !== "" ? Number(r.Unemployment) : null,
          "Inflation Rate": r["Inflation Rate"] !== "" ? Number(r["Inflation Rate"]) : null,
        }));
        setRows(coerced);
        const years = coerced.map((r) => r.Year).filter((y) => !isNaN(y));
        if (years.length) {
          const min = Math.min(...years);
          const max = Math.max(...years);
          setYearBounds([min, max]);
          setSelectedYear(min);
        }
        setProcessing(false);
      },
      error: (err) => {
        console.error(err);
        setProcessing(false);
        alert("Failed to parse CSV. Check headers and numeric fields.");
      },
    });
  };

  const hierarchyData = useMemo(() => {
    if (!rows.length) return null;
    return buildHierarchy(rows, selectedYear);
  }, [rows, selectedYear]);


  // Render Voronoi treemap
  useEffect(() => {
    const svgEl = d3.select(svgRef.current);
    svgEl.selectAll("*").remove();
    if (!hierarchyData) return;

    const { w, h } = dims;
    const svg = svgEl.attr("viewBox", `0 0 ${w} ${h}`);

    function seededRandom(seed) {
      let s = seed;
      return function() {
        s = Math.sin(s) * 10000;
        return s - Math.floor(s);
      };
    }

    
    // Build hierarchy and compute polygons
    const root = d3
      .hierarchy(hierarchyData)
      .sum((d) => d.value || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    const vt = voronoiTreemap()
      .clip([
        [0, 0],
        [0, h],
        [w, h],
        [w, 0],
      ])
      .convergenceRatio(0.015)
      .maxIterationCount(80)
      .minWeightRatio(0.002)
      .prng(seededRandom(12345));

    vt(root);

    // COUNTRY GROUPS (depth === 1)
    const countries = root.children || [];
    const gCountries = svg.append("g").attr("class", "countries");


    //
    countries.forEach((node) => {
      const country = node.data;
      const polygon = node.polygon;
      if (!polygon) return;

      const borderColor = getInflationColor(country.inflation);
      const opacity = getOpacity(country.unemployment);
      const area = Math.abs(d3.polygonArea(polygon));
      const centroid = d3.polygonCentroid(polygon);

      const g = gCountries.append("g").attr("class", "country");
      
      
     

      // Fill:
      if (displayMode === "name" || !(node.children && node.children.length)) {
        g.append("path")
          .attr("d", `M${polygon.join("L")}Z`)
          .attr("fill", continentColors[country.continent] || "#ccc")
          .attr("opacity", opacity)
          .attr("stroke", borderColor)
          .attr("stroke-width", 2)

          //checks if user hovers over a node in tree map if so display tool tip that was referebced in the toolTipRef along with various data
          .on("mouseover", (event)=>{ toolTipRef.current.html(`<strong>${node.data.name}</strong><br/>
                                                    GDP: $${numberConversion(node.value).toLocaleString()}<br/>
                                                    GDP Per Capita: $${node.data.gpdpercapita ?? "N/A"}<br/>
                                                    Agriculture(% GDP): ${node.data.agriculture ?? "N/A"}%<br/>
                                                    Service(% GDP): ${node.data.service?? "N/A"}%<br/>
                                                    Industry(% GDP): ${node.data.industry?? "N/A"}%<br/>
                                                    Inflation Rate: ${node.data.inflation ?? "N/A"}%<br/>
                                                    Unemployment Rate: ${node.data.unemployment ?? "N/A"}%<br/>`)
           
          .style("opacity",1);   //make the tooltip visible                                       
          })
          .on("mousemove", (event)=>{ toolTipRef.current                               // moves tooltip with mouse so mouse doesnt obstruct it
                                      .style("left",event.pageX + 10 +"px")
                                      .style("top", event.pageY + 10 + "px");                                       
          })
          //once user stops hovering over node, make tool tip disappear
          .on("mouseleave", ()=>{
            toolTipRef.current.style("opacity", 0);
          });

      } else {
        // Sub-cells: depth === 2 (components)
        // Draw components first, then a thin outline for the country
        (node.children || []).forEach((compNode) => {
          const compPoly = compNode.polygon;
          if (!compPoly) return;
          g.append("path")
            .attr("d", `M${compPoly.join("L")}Z`)
            .attr("fill", gdpComponentColors[compNode.data.name] || "#ddd")
            .attr("opacity", opacity)
            .attr("stroke", "rgba(0,0,0,0.05)")
            .attr("stroke-width", 1);
            
        });

        // Country outline on top
        g.append("path")
          .attr("d", `M${polygon.join("L")}Z`)
          .attr("fill", "none")
          .attr("stroke", borderColor)
          .attr("stroke-width", 2);
      }

      // Labels (country name + GDP) — size-aware to avoid clutter
      if (area > 1200) {
        g.append("text")
          .attr("x", centroid[0])
          .attr("y", centroid[1])
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("fill", displayMode === "name" ? "#fff" : "#111")
          .style("font-weight", "700")
          .style("font-size", `${Math.max(10, Math.sqrt(area) / 18)}px`)
          .text(country.name);

        if (area > 4200) {
          g.append("text")
            .attr("x", centroid[0])
            .attr("y", centroid[1] + Math.sqrt(area) / 18)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "hanging")
            .attr("fill", displayMode === "name" ? "#fff" : "#333")
            .style("font-size", `${Math.max(9, Math.sqrt(area) / 24)}px`)
            .text(
              (() => {
                const nodeValue = node.value || 0;
                const trillions = nodeValue / 1e12;
                return `$${trillions >= 0.1 ? trillions.toFixed(2) + "T" : (nodeValue / 1e9).toFixed(0) + "B"}`
              })()
            );
        }
      }
    });
  }, [hierarchyData, dims, displayMode, selectedYear]);

  return (

    <div className="w-full min-h-screen bg-white">
      <div className ="body">
      <div className="max-w-7xl mx-auto p-6">
        <h1 className="text-3xl font-bold mb-6 text-center">GDP Voronoi Treemap Visualization</h1>

        {/* Controls */}
        <div className="bg-gray-100 rounded-lg p-4 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium mb-2">Upload CSV File</label>
              <div className="relative">
                <input type="file" accept=".csv" onChange={onFile} className="hidden" id="file-upload" />
                <label
                  htmlFor="file-upload"
                  className="flex items-center justify-center gap-2 bg-blue-500 text-white px-4 py-2 rounded cursor-pointer hover:bg-blue-600"
                >
                  <Upload size={20} />
                  {rows.length > 0 ? "Change File" : "Upload CSV"}
                </label>
              </div>
            </div>

            {/* Year Slider */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Year: {selectedYear}
              </label>
              <input
                type="range"
                min={yearBounds[0]}
                max={yearBounds[1]}
                value={selectedYear}
                onChange={(e) => setSelectedYear(Number(e.target.value))}
                className="w-full"
                disabled={!rows.length}
              />
            </div>

            {/* Display Mode */}
            <div>
              <label className="block text-sm font-medium mb-2">Display Mode</label>
              <select
                value={displayMode}
                onChange={(e) => setDisplayMode(e.target.value)}
                className="w-full px-4 py-2 border rounded"
                disabled={!rows.length}
              >
                <option value="name">Country Name (continent color)</option>
                <option value="makeup">GDP Makeup (subdivided)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="bg-gray-100 rounded-lg p-4 mb-6">
          <h3 className="font-bold mb-3">Legend</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div><strong>Size:</strong> GDP (area)</div>
            <div><strong>Opacity:</strong> Employment rate</div>
            <div><strong>Border:</strong> Inflation (Green=+, Red=-, White=NULL)</div>
            <div><strong>Color:</strong> {displayMode === "name" ? "Continent" : "GDP Components"}</div>
          </div>
          <div className="flex flex-wrap gap-3 mt-3 text-sm">
            {displayMode === "name"
              ? Object.entries(continentColors).map(([k, v]) => (
                  <div key={k} className="flex items-center gap-2">
                    <span className="inline-block w-4 h-4 rounded" style={{ background: v }} />
                    {k}
                  </div>
                ))
              : Object.entries(gdpComponentColors).map(([k, v]) => (
                  <div key={k} className="flex items-center gap-2">
                    <span className="inline-block w-4 h-4 rounded" style={{ background: v }} />
                    {k}
                  </div>
                ))}
          </div>
        </div>

        {/* Chart */}
        <div ref={wrapperRef} className="w-full bg-white rounded-lg shadow-sm ring-1 ring-gray-200">
          <svg ref={svgRef} className="w-full h-auto block" />
          {!rows.length && (
            <div className="text-center py-20">
              <Upload size={48} className="mx-auto mb-4 text-gray-400" />
              <div className="text-xl text-gray-600">Upload a CSV file to begin</div>
            </div>
          )}
        </div>

        {processing && (
          <div className="text-center py-6 text-gray-600">Processing data…</div>
        )}
      </div>
      </div>
    </div>
  );
};

export default VoronoiTreemap;
