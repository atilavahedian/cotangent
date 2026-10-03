import React from "react";
import {DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative,
  SortableItem, SortableRegion, useDataApp} from "../../data-app-public.jsx";

const decimal = (digits) => (value) => value == null ? "Censored" : Number(value).toFixed(digits);
const summaryColumns = [
  {field:"method", label:"Backward method"},
  {field:"test_bpb", label:"Test BPB ↓", renderCell:decimal(4)},
  {field:"target_seconds", label:"Time to 3.2 BPB (s) ↓", renderCell:decimal(2)},
  {field:"target_status", label:"Target reached"},
  {field:"training_seconds", label:"2,000 steps (s) ↓", renderCell:decimal(2)},
];
const pairColumns = [
  {field:"seed", label:"Seed"},
  {field:"native_target_seconds", label:"Native time (s)", renderCell:decimal(2)},
  {field:"adaptive_target_seconds", label:"Cotangent time (s)", renderCell:decimal(2)},
  {field:"elapsed_reduction_fraction", label:"Time saved (%)", renderCell:(v)=>v==null?"Censored":(100*v).toFixed(2)},
  {field:"test_difference_bpb", label:"Test Δ BPB", renderCell:decimal(4)},
];
const pilotColumns = [
  {field:"method", label:"Method"},
  {field:"training_seconds", label:"120-step time (s)", renderCell:decimal(3)},
  {field:"diagnostic_bpb", label:"Train holdout BPB", renderCell:decimal(4)},
];
const order=["findings", "curves", "controller", "mathematics", "study-design", "pilots", "limits", "reproduce"];

export function ReportContent() {
  const {snapshot, reviewedPeriodRows, visible, canEdit, mode, appTitle, setAppTitle} = useDataApp();
  const copy=snapshot.report?.narratives ?? {};
  const final=!!snapshot.queries?.method_summary;
  const rows=(query)=>reviewedPeriodRows(query);
  const summaries=final?rows("method_summary").filter(r=>r.group==="small"):[];
  const pairs=final?rows("paired_results").filter(r=>r.group==="small"):[];
  const curves=final?rows("learning_curves").filter(r=>r.group==="small"):[];
  const fractions=final?rows("sampling_curves").filter(r=>r.group==="small"):[];
  const pilots=rows("pilot_comparison");
  const narrative=(id,title,query,source)=> <ReportSection id={id} title={title}
    queryId={query} sourceRows={source} showHeading={false}>
    <RichNarrative id={`cotangent:${id}`} value={copy[id]??""} label={`Edit ${title}`} />
  </ReportSection>;
  return <article className="report-content" aria-label="Cotangent research report">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode==="edit"} suppressContentEditableWarning
        aria-label={canEdit && mode==="edit"?"Edit report heading":undefined}
        onBlur={canEdit && mode==="edit"?(e)=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined}
        onKeyDown={canEdit && mode==="edit"?(e)=>{if(e.key==="Enter"){e.preventDefault();e.currentTarget.blur();}}:undefined}>{appTitle}</h1>
      <RichNarrative id="cotangent:deck" value={copy.deck??"A new, measured investigation into cheaper transformer backpropagation."} className="report-deck" label="Edit introduction" />
    </header>
    <SortableRegion id="cotangent:sections" label="Research report sections" variant="stack" authoredOrder={order} className="report-sortable-sections">
      {visible("findings") && <SortableItem id="findings" label="Result" kind="narrative">
        <section className="report-section">
          {narrative("findings","Result",final?"paired_results":"pilot_comparison",final?pairs:pilots)}
          {final && <DataComponent id="method-table" title="Five-seed comparison · 3.35M parameters" queryId="method_summary" kind="table" sourceRows={summaries} displayRows={summaries}>
            <DataTable rows={summaries} columns={summaryColumns} searchable={false} compactNumbers={false} label="Mean final quality and elapsed time to target across five seeds" />
          </DataComponent>}
          {final && <DataComponent id="paired-table" title="Every primary pair" queryId="paired_results" kind="table" sourceRows={pairs} displayRows={pairs}>
            <DataTable rows={pairs} columns={pairColumns} searchable={false} compactNumbers={false} label="Five paired seeds, native versus Cotangent" />
          </DataComponent>}
        </section>
      </SortableItem>}
      {final && visible("curves") && <SortableItem id="curves" label="Learning curves" kind="chart">
        <section className="report-section">
          {narrative("curves","Learning curves","learning_curves",curves)}
          <EvidenceChart id="quality-curve" queryId="learning_curves" title="Mean validation quality by optimizer step"
            spec={{type:"line",x:"step",y:"validation_bpb",series:"method",stackable:false,startAtZero:false,valueDecimals:3,xLabel:"Optimizer step",yLabel:"Bits per byte · lower is better",annotations:[{id:"frozen-target",kind:"benchmark",measure:"validation_bpb",field:"target_bpb",label:"Frozen target: 3.2 BPB"}]}}
            rows={curves} sourceRows={curves} height={380} />
          <EvidenceChart id="elapsed-curve" queryId="learning_curves" title="Native and Cotangent: quality against elapsed time"
            spec={{type:"line",x:"elapsed_seconds",y:"validation_bpb",series:"method",stackable:false,startAtZero:false,valueDecimals:3,xLabel:"Elapsed seconds, including setup and evaluations",yLabel:"Bits per byte · lower is better",annotations:[{id:"frozen-target",kind:"benchmark",measure:"validation_bpb",field:"target_bpb",label:"Frozen target: 3.2 BPB"}]}}
            rows={curves.filter(r=>["native","adaptive"].includes(r.mode))} sourceRows={curves.filter(r=>["native","adaptive"].includes(r.mode))} height={380} />
        </section>
      </SortableItem>}
      {visible("controller") && <SortableItem id="controller" label="Adaptive computation" kind="narrative">
        <section className="report-section">
          {narrative("controller","Adaptive computation",final?"sampling_curves":"pilot_comparison",final?fractions:pilots)}
          {final && <EvidenceChart id="sample-curve" queryId="sampling_curves" title="How much of the weight-gradient row budget was used?"
            spec={{type:"line",x:"step",y:"sample_fraction",series:"seed",stackable:false,valueDecimals:3,xLabel:"Optimizer step",yLabel:"Logged mean layer fraction · 1 = exact"}}
            rows={fractions} sourceRows={fractions} height={340} />}
        </section>
      </SortableItem>}
      {visible("mathematics") && <SortableItem id="mathematics" label="Mathematical method" kind="narrative">
        {narrative("mathematics","Mathematical method",final?"integrity":"pilot_comparison",final?rows("integrity"):pilots)}
      </SortableItem>}
      {visible("study-design") && <SortableItem id="study-design" label="Experiment design" kind="narrative">
        {narrative("study-design","Experiment design",final?"integrity":"pilot_comparison",final?rows("integrity"):pilots)}
      </SortableItem>}
      {visible("pilots") && <SortableItem id="pilots" label="Development record" kind="table">
        <section className="report-section">
          {narrative("pilots","Development record","pilot_comparison",pilots)}
          <DataComponent id="pilot-table" title="Diagnostic pilots · one seed, training-only holdout" queryId="pilot_comparison" kind="table" sourceRows={pilots} displayRows={pilots}>
            <DataTable rows={pilots} columns={pilotColumns} searchable={false} compactNumbers={false} label="Development-only pilot measurements, not final results" />
          </DataComponent>
        </section>
      </SortableItem>}
      {visible("limits") && <SortableItem id="limits" label="Scope and limitations" kind="narrative">
        {narrative("limits","Scope and limitations",final?"method_summary":"pilot_comparison",final?rows("method_summary"):pilots)}
      </SortableItem>}
      {visible("reproduce") && <SortableItem id="reproduce" label="Reproduce the work" kind="narrative">
        {narrative("reproduce","Reproduce the work",final?"integrity":"pilot_comparison",final?rows("integrity"):pilots)}
      </SortableItem>}
    </SortableRegion>
  </article>;
}
