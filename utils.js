// utils.js — helper: robust CSV parse, numeric matrix, normalize, example generator, logger
const Utils = (function(){
  function parseCSVText(text){
    // robust parse: handle quoted commas
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    if(lines.length === 0) return {headers:[], rows:[]};

    function parseLine(line){
      const cells = []; let cur=''; let inQuotes=false;
      for(let i=0;i<line.length;i++){
        const ch = line[i];
        if(ch === '"'){
          if(inQuotes && line[i+1]==='"'){ cur += '"'; i++; continue; }
          inQuotes = !inQuotes;
        } else if(ch === ',' && !inQuotes){
          cells.push(cur);
          cur = '';
        } else cur += ch;
      }
      cells.push(cur);
      return cells.map(c => c.trim());
    }

    const headers = parseLine(lines[0]);
    const rows = lines.slice(1).map(line=>{
      const cols = parseLine(line);
      while(cols.length < headers.length) cols.push('');
      const obj = {};
      headers.forEach((h,i)=> obj[h]=cols[i] ?? '');
      return obj;
    });
    return {headers, rows};
  }

  function toNumericMatrix(rows, featureNames, labelName){
    const X = []; const y = [];
    rows.forEach(r=>{
      const xr = featureNames.map(f => {
        const v = r[f];
        const n = Number(v);
        return Number.isFinite(n) ? n : NaN;
      });
      const lab = Number(r[labelName]);
      if(xr.some(v=>Number.isNaN(v)) || Number.isNaN(lab)) return; // skip bad row
      X.push(xr); y.push(lab);
    });
    return {X,y};
  }

  function normalize(X){
    if(!X || X.length===0) return {Xn:[], mins:[], maxs:[]};
    const cols = X[0].length;
    const mins = Array(cols).fill(Infinity), maxs = Array(cols).fill(-Infinity);
    X.forEach(r=> r.forEach((v,i)=>{ mins[i]=Math.min(mins[i],v); maxs[i]=Math.max(maxs[i],v); }));
    const Xn = X.map(r=> r.map((v,i)=> (maxs[i]===mins[i]) ? 0.5 : (v - mins[i])/(maxs[i]-mins[i]) ));
    return {Xn, mins, maxs};
  }

  function sampleExampleCSV(){
    const header = 'age,pressure,flow,leak_history,corrosion_index,failure\n';
    let lines = '';
    for(let i=0;i<1000;i++){
      const age = (Math.random()*60+0.1).toFixed(1);
      const pressure = (Math.random()*7+1).toFixed(2);
      const flow = (Math.random()*495+5).toFixed(2);
      const leak_history = Math.random()>0.88 ? 1 : 0;
      const corrosion_index = (Math.random()*10).toFixed(2);
      const score = (age/60)*0.4 + (corrosion_index/10)*0.35 + leak_history*0.18 + Math.abs(pressure-4)/4*0.07;
      const failure = Math.random() < Math.min(0.95, score) ? 1 : 0;
      lines += [age,pressure,flow,leak_history,corrosion_index,failure].join(',') + '\n';
    }
    return header + lines;
  }

  function log(msg){
    const pre = document.getElementById('consoleOut');
    if(!pre) return;
    const time = new Date().toLocaleTimeString();
    pre.textContent += `[${time}] ${msg}\n`;
    pre.scrollTop = pre.scrollHeight;
    console.log(msg);
  }

  return {parseCSVText, toNumericMatrix, normalize, sampleExampleCSV, log};
})();
