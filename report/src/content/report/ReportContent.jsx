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
const packingSummaryColumns = [
  {field:"size", label:"Model"},
  {field:"method", label:"Training implementation"},
  {field:"seeds", label:"Seeds"},
  {field:"test_bpb", label:"Test BPB ↓", renderCell:decimal(4)},
  {field:"target_seconds", label:"3.2 BPB (s) ↓", renderCell:decimal(2)},
  {field:"training_seconds", label:"Fixed-budget training (s) ↓", renderCell:decimal(2)},
];
const packingPairColumns = [
  {field:"size", label:"Model"},
  {field:"seed", label:"Seed"},
  {field:"native_target_seconds", label:"Native fused (s)", renderCell:decimal(2)},
  {field:"packed_target_seconds", label:"Cotangent (s)", renderCell:decimal(2)},
  {field:"elapsed_reduction_fraction", label:"Time saved (%)", renderCell:(v)=>v==null?"Censored":(100*v).toFixed(2)},
  {field:"test_difference_bpb", label:"Test Δ BPB", renderCell:decimal(4)},
];
const order=["batching-findings", "batching-method", "batching-design", "deterministic-findings", "deterministic-method", "deterministic-design", "packing-findings", "packing-method", "packing-design", "findings", "curves", "controller", "mathematics", "study-design", "pilots", "limits", "reproduce"];

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
  const packing=!!snapshot.queries?.packing_summary;
  const packingSummaries=packing?rows("packing_summary"):[];
  const packingPairs=packing?rows("packing_pairs"):[];
  const packingCurves=packing?rows("packing_curves"):[];
  const deterministic=!!snapshot.queries?.deterministic_summary;
  const deterministicSummaries=deterministic?rows("deterministic_summary"):[];
  const deterministicPairs=deterministic?rows("deterministic_pairs"):[];
  const deterministicAblation=deterministic?rows("deterministic_ablation"):[];
  const deterministicCurves=deterministic?rows("deterministic_curves"):[];
  const batching=!!snapshot.queries?.batching_summary;
  const batchingSummaries=batching?rows("batching_summary").filter(r=>["native-fused","native-bucket"].includes(r.arm)):[];
  const batchingPairs=batching?rows("batching_pairs"):[];
  const batchingEquivalence=batching?rows("batching_equivalence"):[];
  const batchingCurves=batching?rows("batching_curves"):[];
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
      {batching && visible("batching-findings") && <SortableItem id="batching-findings" label="V4: native backward and norm batching" kind="chart">
        <section className="report-section">
          {narrative("batching-findings","V4: native backward and norm batching","batching_pairs",batchingPairs)}
          <DataComponent id="batching-method-table" title="V4 · unchanged native backward at two model sizes" queryId="batching_summary" kind="table" sourceRows={batchingSummaries} displayRows={batchingSummaries}>
            <DataTable rows={batchingSummaries} columns={packingSummaryColumns} searchable={false} compactNumbers={false} label="Twenty-five primary and three descriptive pairs against native fused AdamW" />
          </DataComponent>
          <DataComponent id="batching-paired-table" title="Every V4 native-backprop pair · all seeds retained" queryId="batching_pairs" kind="table" sourceRows={batchingPairs} displayRows={batchingPairs}>
            <DataTable rows={batchingPairs} columns={packingPairColumns} searchable={false} compactNumbers={false} label="All 25 primary and three larger-model descriptive pairs" />
          </DataComponent>
          <DataComponent id="batching-equivalence-table" title="Seven full-training update-equivalence pairs" queryId="batching_equivalence" kind="table" sourceRows={batchingEquivalence} displayRows={batchingEquivalence}>
            <DataTable rows={batchingEquivalence} columns={[
              {field:"size",label:"Model"},{field:"seed",label:"Seed"},
              {field:"identical_final_model_hash",label:"Identical final weights",renderCell:v=>v?"Yes":"No"},
              {field:"test_difference_bpb",label:"Full-test Δ BPB",renderCell:decimal(8)},
              {field:"elapsed_reduction_fraction",label:"Time saved (%)",renderCell:v=>v==null?"Censored":(100*v).toFixed(2)},
            ]} searchable={false} compactNumbers={false} label="Five small and two larger pairs with the same deterministic cotangents in both arms" />
          </DataComponent>
          {["small","medium"].map(size=> <EvidenceChart key={size} id={`batching-${size}-curve`} queryId="batching_curves"
            title={size==="small"?"V4 · 3.35M parameters, 25 primary pairs":"V4 · 19.28M parameters, three descriptive pairs"}
            spec={{type:"line",x:"elapsed_seconds",y:"validation_bpb",series:"method",stackable:false,startAtZero:false,valueDecimals:3,xLabel:"Elapsed seconds, including setup and probes",yLabel:"Validation bits per byte · lower is better"}}
            rows={batchingCurves.filter(r=>r.size===size && ["native-fused","native-bucket"].includes(r.arm))} sourceRows={batchingCurves.filter(r=>r.size===size && ["native-fused","native-bucket"].includes(r.arm))} height={360} />)}
        </section>
      </SortableItem>}
      {batching && visible("batching-method") && <SortableItem id="batching-method" label="V4: numerical method" kind="narrative">
        {narrative("batching-method","V4: numerical method","batching_integrity",rows("batching_integrity"))}
      </SortableItem>}
      {batching && visible("batching-design") && <SortableItem id="batching-design" label="V4: independent freeze" kind="narrative">
        {narrative("batching-design","V4: independent freeze","batching_integrity",rows("batching_integrity"))}
      </SortableItem>}
      {deterministic && visible("deterministic-findings") && <SortableItem id="deterministic-findings" label="V3: deterministic exact gradients" kind="chart">
        <section className="report-section">
          {narrative("deterministic-findings","V3: deterministic exact gradients","deterministic_pairs",deterministicPairs)}
          <DataComponent id="deterministic-method-table" title="V3 · all 41 frozen runs" queryId="deterministic_summary" kind="table" sourceRows={deterministicSummaries} displayRows={deterministicSummaries}>
            <DataTable rows={deterministicSummaries} columns={packingSummaryColumns} searchable={false} compactNumbers={false} label="V3 primary, larger-model descriptive and deterministic-control means" />
          </DataComponent>
          <DataComponent id="deterministic-paired-table" title="Every V3 primary and larger-model pair" queryId="deterministic_pairs" kind="table" sourceRows={deterministicPairs} displayRows={deterministicPairs}>
            <DataTable rows={deterministicPairs} columns={packingPairColumns} searchable={false} compactNumbers={false} label="Fifteen primary pairs and three descriptive larger-model pairs; all seeds retained" />
          </DataComponent>
          <DataComponent id="deterministic-ablation-table" title="Five descriptive ablations · control uses the same deterministic embedding" queryId="deterministic_ablation" kind="table" sourceRows={deterministicAblation} displayRows={deterministicAblation}>
            <DataTable rows={deterministicAblation} columns={packingPairColumns.map(c=>c.field==="native_target_seconds"?{...c,label:"Deterministic control (s)"}:c)} searchable={false} compactNumbers={false} label="Packing versus deterministic-incidence native optimizer on five predeclared seeds" />
          </DataComponent>
          {["small","medium"].map(size=> <EvidenceChart key={size} id={`deterministic-${size}-curve`} queryId="deterministic_curves"
            title={size==="small"?"V3 · 3.35M parameters, 15 primary pairs":"V3 · 19.28M parameters, three descriptive pairs"}
            spec={{type:"line",x:"elapsed_seconds",y:"validation_bpb",series:"method",stackable:false,startAtZero:false,valueDecimals:3,xLabel:"Elapsed seconds, including setup and probes",yLabel:"Validation bits per byte · lower is better"}}
            rows={deterministicCurves.filter(r=>r.size===size && r.arm!=="incidence-fused")} sourceRows={deterministicCurves.filter(r=>r.size===size && r.arm!=="incidence-fused")} height={360} />)}
        </section>
      </SortableItem>}
      {deterministic && visible("deterministic-method") && <SortableItem id="deterministic-method" label="V3: exact cotangents and diagnosis" kind="narrative">
        {narrative("deterministic-method","V3: exact cotangents and diagnosis","deterministic_integrity",rows("deterministic_integrity"))}
      </SortableItem>}
      {deterministic && visible("deterministic-design") && <SortableItem id="deterministic-design" label="V3: independent frozen protocol" kind="narrative">
        {narrative("deterministic-design","V3: independent frozen protocol","deterministic_integrity",rows("deterministic_integrity"))}
      </SortableItem>}
      {visible("packing-findings") && <SortableItem id="packing-findings" label="V2: exact packing results" kind="chart">
        <section className="report-section">
          {narrative("packing-findings","V2: exact packing results",packing?"packing_pairs":"pilot_comparison",packing?packingPairs:pilots)}
          {packing && <DataComponent id="packing-method-table" title="V2 · stronger baseline, two model sizes" queryId="packing_summary" kind="table" sourceRows={packingSummaries} displayRows={packingSummaries}>
            <DataTable rows={packingSummaries} columns={packingSummaryColumns} searchable={false} compactNumbers={false} label="Exact packing against native fused AdamW, all 24 frozen runs" />
          </DataComponent>}
          {packing && <DataComponent id="packing-paired-table" title="Every V2 pair · no discarded seeds" queryId="packing_pairs" kind="table" sourceRows={packingPairs} displayRows={packingPairs}>
            <DataTable rows={packingPairs} columns={packingPairColumns} searchable={false} compactNumbers={false} label="Seven small-model and five larger-model paired outcomes" />
          </DataComponent>}
          {packing && ["small","medium"].map(size=> <EvidenceChart key={size} id={`packing-${size}-curve`} queryId="packing_curves"
            title={size==="small"?"V2 · 3.35M parameters, seven seeds":"V2 · 19.28M parameters, five seeds"}
            spec={{type:"line",x:"elapsed_seconds",y:"validation_bpb",series:"method",stackable:false,startAtZero:false,valueDecimals:3,xLabel:"Elapsed seconds, including setup and probes",yLabel:"Validation bits per byte · lower is better"}}
            rows={packingCurves.filter(r=>r.size===size)} sourceRows={packingCurves.filter(r=>r.size===size)} height={360} />)}
        </section>
      </SortableItem>}
      {visible("packing-method") && <SortableItem id="packing-method" label="V2: exact update equivalence" kind="narrative">
        {narrative("packing-method","V2: exact update equivalence",packing?"packing_integrity":"pilot_comparison",packing?rows("packing_integrity"):pilots)}
      </SortableItem>}
      {visible("packing-design") && <SortableItem id="packing-design" label="V2: frozen experiment" kind="narrative">
        {narrative("packing-design","V2: frozen experiment",packing?"packing_integrity":"pilot_comparison",packing?rows("packing_integrity"):pilots)}
      </SortableItem>}
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
