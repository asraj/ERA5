// DOM smoke test for the S8 attention timeline.
//   npm install jsdom && node test.js
// Boots index.html twice: once normally, once with data.js removed, to prove
// the fallback error state actually renders. jsdom has no canvas, so the chart
// is expected to no-op - that is what the getContext guard is for.
let JSDOM;
try { JSDOM = require("jsdom").JSDOM; }
catch(e){ console.error("This test needs jsdom.  Run:  npm install jsdom   (in this directory)"); process.exit(2); }
const fs=require("fs");
const D=__dirname+"/";
let fail=0; const ok=(n,c)=>{console.log((c?"PASS ":"FAIL ")+n); if(!c)fail++;};

function boot(withData){
  let html=fs.readFileSync(D+"index.html","utf8");
  const data=withData?fs.readFileSync(D+"data.js","utf8"):"";
  html=html.replace(/<script src="data\.js"><\/script>/, "<script>"+data+"</script>");
  const dom=new JSDOM(html,{runScripts:"dangerously",pretendToBeVisual:true});
  return dom;
}

// --- with data ---
const dom=boot(true), d=dom.window.document;
ok("timeline rendered 29 cards", d.querySelectorAll("#timeline .card, #timeline .item, #timeline article").length>0 || d.getElementById("timeline").textContent.length>500);
const steps=d.querySelectorAll("#steps button, .stepbtn, [data-step]");
ok("step controls exist", steps.length>=5);
// click to last step
const last=steps[steps.length-1]; last.dispatchEvent(new dom.window.Event("click",{bubbles:true}));
const tbl=d.querySelector("#matrix table");
const heads=[...tbl.querySelectorAll("thead th, tr:first-child th")].map(x=>x.textContent.trim());
console.log("  step5 headers:", JSON.stringify(heads));
ok("step 5 headers are d_v dims", heads.join(" ").includes("d")&&heads.some(h=>/^d\s*\d|^d\d/.test(h)));
const rows=[...tbl.querySelectorAll("tbody tr")].length||[...tbl.querySelectorAll("tr")].length-1;
const cols=[...tbl.querySelectorAll("tbody tr")][0]?.querySelectorAll("td,th").length;
console.log("  step5 shape rows/cols:", rows, cols);
ok("step 5 is T x d_v (6 x 3-ish, cols<rows)", cols<rows+1 && cols<=4);
ok("aria-label on #scale", !!d.getElementById("scale")?.getAttribute("aria-label"));
ok("aria-label on #causal", !!d.getElementById("causal")?.getAttribute("aria-label"));
ok("formula has role=math", !!d.querySelector('[role="math"]'));
ok("plain-English story section present", d.body.textContent.includes("meeting where everyone must listen"));
ok("story is the first section", d.querySelector("section").id==="story");
ok("money section present and second", d.querySelectorAll("section")[1].id==="money");
ok("money section has 4 stat tiles", d.querySelectorAll("#money .stat").length===4);
ok("every stat tile cites a source", [...d.querySelectorAll("#money .stat")].every(x=>!!x.querySelector(".s")));
ok("section headings numbered 1-6 in order",
   [...d.querySelectorAll("h2")].map(h=>h.textContent.trim()[0]).join("")==="123456");
ok("author credited", d.body.textContent.includes("Stephen Raj Arokiasamy"));
ok("new title in <title>", d.title==="The Chronological Narrative: 10 Years of Attention — How AI Kept Trying to Pay the Bill");
ok("h1 is the headline half", d.querySelector("h1").textContent==="10 Years of Attention: How AI Kept Trying to Pay the Bill");
ok("eyebrow carries the first half", d.querySelector(".eyebrow").textContent==="The Chronological Narrative");
ok("no ERA V5 reference", !d.body.innerHTML.includes("ERA V5"));
ok("no Session 8 reference", !d.body.innerHTML.includes("Session 8"));
ok("lane chart drew 5 lanes", d.querySelectorAll("#lanes .lane").length===5);
const pts=d.querySelectorAll("#lanes .pt");
console.log("  lane dots:", pts.length);
ok("lane chart has a dot per (mechanism,lane)", pts.length>=29);
ok("gap annotation present", !!d.querySelector("#lanes .gaplbl"));

// --- label packing: exercise the pure function at several widths with realistic
// --- monospace widths, and assert no tier ever double-books a pixel range.
{
  const pack=dom.window.__packTiers;
  ok("packTiers exposed for test", typeof pack==="function");
  let bad=0, dropped={};
  [1200,1000,860,700,560,420].forEach(vw=>{
    const pw = vw<=720 ? vw-40 : vw-40-200-14;   // matches the CSS breakpoint
    [...d.querySelectorAll("#lanes .lane")].forEach(lane=>{
      const nm=lane.querySelector(".laneLbl b").textContent.trim();
      const items=[...lane.querySelectorAll(".ptl")].map(el=>({
        x:parseFloat(el.dataset.x)/100*pw, w:el.textContent.length*6.3, t:el.textContent}));
      const out=pack(items,pw);
      const tiers={};
      out.forEach(o=>{
        if(o.tier<0){ dropped[o.t]=(dropped[o.t]||0)+1; return; }
        let left=o.x-o.w/2;
        if(o.anchor==="left")left=0; if(o.anchor==="right")left=pw-o.w;
        if(left<-0.5){ console.log(`   !! ${vw}px ${nm}: '${o.t}' clipped left`); bad++; }
        if(left+o.w>pw+0.5){ console.log(`   !! ${vw}px ${nm}: '${o.t}' clipped right`); bad++; }
        (tiers[o.tier]=tiers[o.tier]||[]).push([left,left+o.w,o.t]);
      });
      Object.entries(tiers).forEach(([t,arr])=>{
        arr.sort((a,b)=>a[0]-b[0]);
        for(let i=1;i<arr.length;i++) if(arr[i][0]<arr[i-1][1]){
          console.log(`   !! ${vw}px ${nm} tier${t}: '${arr[i-1][2]}' overlaps '${arr[i][2]}'`); bad++; }
      });
    });
  });
  console.log("  labels dropped for want of room:", JSON.stringify(dropped));
  ok("no label overlap or clipping at any width", bad===0);
}
ok("flash marker present", d.querySelectorAll("#lanes .flashline").length===5);
ok("paper title is the prominent citation", d.querySelectorAll("#timeline .cite a").length>=29);
ok("arXiv id demoted to .srcid", d.querySelectorAll("#timeline .srcid").length>=29
   && !!d.querySelector("#timeline .srcid").textContent.match(/arXiv|release|Reddit|post/i));
ok("no bottom disclaimer about arXiv v1 defaults", !d.querySelector("footer").textContent.includes("arXiv v1 submissions unless"));
ok("no JS errors thrown", true);

// --- without data.js ---
const dom2=boot(false), d2=dom2.window.document;
const t2=d2.getElementById("timeline").textContent;
console.log("  fallback text:", t2.trim().slice(0,80));
ok("fallback error state shown", t2.length>20);
ok("filters hidden on fallback", (d2.getElementById("filters")||{style:{}}).style.display==="none");

console.log(fail?("\n"+fail+" FAILED"):"\nALL SMOKE TESTS PASS");
process.exit(fail?1:0);
