"""Freeze task-local profile/images/prompts; CLI dump/help only, no inference."""
import hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='d7b1d37ff0e00c0011b324eef309534d01947e44'
TEMP=Path(tempfile.mkdtemp(prefix='ds41f-step3br3-session-'))
for d in ['workspace','agents','runtime','home/profiles/headless']:(TEMP/d).mkdir(parents=True)
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
save('location.json',dict(temp=str(TEMP)))
old=ROOT/'artifacts/multimodal-runtime/phase3-step3a-evidence'
manifest=json.loads((old/'images.json').read_text())[:2]
for row in manifest:
 source=old/('image-'+row['label']+'.png');dest=TEMP/'workspace'/source.name
 assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
 shutil.copyfile(source,dest);dest.chmod(0o444);row['copy']=str(dest)
save('images.json',manifest)
profile=TEMP/'home/profiles/headless'
(profile/'package.json').write_text(json.dumps(dict(name='ds41f-step3br3',private=True,dependencies={},dsh=dict(profile=dict(bundles=['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless'])))))
(profile/'cordis.yml').write_text('[]\n')
patch=(old/'profile.patch.yml').read_text()
patch=re.sub(r'workspaceRoot: .*', 'workspaceRoot: '+str(TEMP/'workspace'),patch)
patch=patch.replace(str(old/'budget-guard.mjs'),str(OUT/'guard.mjs'))
patch+='''
- id: compaction-basic
  config:
    auto: true
    modelPolicies:
      - provider: deepseek-official
        model: deepseek-flash
        thresholdRatio: 0.00390625
        headroomTokens: 2048
        retainTokens: 1024
        maxTokens: 2048
        compactionRetries: 0
'''
(profile/'cordis.patch.yml').write_text(patch);(OUT/'profile.patch.yml').write_text(patch)
prompts=[
'''Inspect image-A.png and image-B.png using only read_image, one tool call per response. Read A first, describe its visible shape, color and exact printed marker before requesting B. After B's actual result, describe B and compare the two. Finish with a useful concise inspection record: distinguish directly observed facts from uncertainty and explain how a reviewer can tell the items apart. Do not infer contents from filenames. Use exactly two read_image calls total, no other tools and no editing. Keep the final record around 250 words.''',
'''Apply this review reference to the two images already inspected; do not read files again. Write a practical 300-word annotated checklist grounded in the actual observations, with one finding per major section, and flag anything not supported by the images.

Reference 1 — Visual evidence and identification. A review record should let another person distinguish what was observed from what was merely expected. Begin with a separate item entry for each source. Record the dominant shape, the dominant fill color, the placement of the graphic relative to its background, and any legible printed identifier. These fields must come from inspection, not from names or an assumed numbering system. Where a field cannot be read confidently, use an explicit unknown rather than guessing. A printed identifier is a transcription task: preserve letter case and digit order, do not silently normalize a similar-looking glyph. Describe uncertainty at the character level if necessary. A source label in our conversation is only a reference to a file and is not itself visual proof.

Separation of evidence classes. Keep a short direct-observation column, an interpretation column, and an unavailable-evidence column. For example, a visible boundary can support a geometric description, while the original vector path, authoring software, or production history is not established by a raster inspection. Avoid inventing material, texture, physical dimensions or lighting conditions for a flat graphic. A uniform-looking background may be described as it appears, but do not claim exact pixel values without a measurement tool. Likewise, qualitative color names are useful communication, not calibrated colorimetry. There is no need for numerical coordinates when an ordinary relative position conveys the observation accurately.

Comparison discipline. Compare corresponding features, not an arbitrary mixture of categories. State a similarity only when it is visible in both sources; state a difference only when it separates the sources rather than restating their labels. Distinguish the main graphic from the printed identifier and background. For each contrast, explain whether shape, color, text, placement or some combination supplies the decisive cue. Do not let one salient cue erase an independent field. A reviewer should still be able to recognize which item was meant if one cue is unavailable, such as when color is not reliably reproduced. If the two graphics have a shared composition, mention that separately from the distinguishing features.

Record structure. Give each item an observation sentence and one caveat sentence. Follow with a compact comparison table or parallel bullets so the same fields occur in the same order. Finish with a single conclusion about identifiability based on the visible evidence. Keep the observed printed identifiers intact in this record, and do not replace them with made-up inventory numbers. Preserve enough information that a later reviewer can audit the comparison without needing a newly invented explanation. This is a bounded descriptive review, not a claim of general image recognition quality or an accessibility certification.

Error handling. If an observation in the earlier exchange conflicts with the visible source, identify the uncertainty rather than quietly rewriting the record. Do not create an extra acquisition or substitute a new image. If no contradiction is evident, say so briefly without claiming an independent remeasurement. Treat the existing reads as the evidence source. Keep recommendations separate from observations, and avoid a generic affirmative checklist that would look the same for any pair of pictures.''',
'''Use the following handoff standard to revise the inspection into a practical 300-word reviewer handoff. No further tools. Apply it specifically to the already inspected pair; retain exact observed identifiers and do not add visual facts that were never observed.

Reference 2 — Review handoff and accessibility. The handoff should explain how a reader can identify an item without relying on color alone. Pair a geometric description with a textual identifier wherever the identifier is genuinely visible. Treat the identifier as observed content, not as a private filesystem path or a guessed meaning. A concise description should include the primary subject, the distinguishing relationship to the companion item, and the text needed for identification. Do not turn alternate text into an exhaustive list of every possible attribute. If the graphic is purely schematic, say what is depicted rather than inventing a real-world object that happens to resemble it.

Reading order. Describe one item fully enough to establish its identity, then describe the other using the same field order. Present shared layout information once if that avoids needless repetition, but do not merge the two identifiers or imply they are interchangeable. A comparison is easier to audit when corresponding fields line up. Keep uncertainty next to the field it qualifies, not in a distant blanket disclaimer. Readers should be able to tell which details are visual observations and which are recommendations for a future publication process. Do not label a recommended correction as if it already happened.

Limits of this review. A rendered picture can support a qualitative account of color and apparent geometry. It does not establish the source document's semantic labels, interactive behavior, font accessibility, contrast ratio, keyboard navigation, or compatibility with a particular assistive device. We have not measured these properties here. Avoid translating an attractive appearance into a compliance claim. If a real application later uses these graphics, its text alternatives and neighboring explanations should be reviewed in that application's actual context. Nothing in this bounded comparison proves universal usability for every reader or device.

Handoff checklist. Include a concise evidence statement for each item, a short set of discriminating cues, and a clear statement of what remains unknown. Separate a suggested plain-language description from the factual audit record. The suggested description should be faithful to the observations and should not contain our internal file paths. The audit record may refer to A and B for traceability. Make clear whether a text identifier is above, below, or within the main graphic only if that placement was observed. Do not estimate exact sizes, coordinates, RGB values, or ratios from memory. Prefer useful relative statements over false precision.

Consistency checks. Verify that the identifier paired with a shape remains the same in the item entry, comparison and handoff. Verify that a shared attribute is not accidentally listed as a difference. Verify that no uncertain glyph has been promoted to a confident transcription merely because it looks plausible. If the earlier reads support the existing transcription clearly, preserve it without inventing an uncertainty. Do not claim another tool pass occurred. Record any limitation honestly while retaining the successful observations. The aim is a usable, source-grounded working record, not a verbose disclaimer or a second acquisition campaign.

Decision note. Conclude with the strongest supported identification strategy for the pair and one bounded caution about what the current evidence cannot establish. Keep that caution relevant to this task. A future reviewer should be able to reuse the record without confusing recommendations, direct visual facts, and untested application behavior.''',
'''Apply this final quality-control reference to the same pair in about 300 words, without any new image reads. Produce an actionable final inspection checklist that can be retained for a later short handoff.

Reference 3 — Quality control and traceable conclusions. Use an explicit distinction between source identity, occurrence of an inspection, and the descriptive conclusion drawn from it. The fact that an item was mentioned repeatedly in discussion does not create another inspection. Refer back to the original two reads; do not claim a fresh verification. A visual description can be refined for clarity, but it must not be quietly expanded into a new factual observation that the earlier evidence never supplied. If a conclusion depends on both items, name the compared attribute and state the evidence from each side.

Audit trail. Preserve the stable A/B labels and the printed markers actually observed. A reader should be able to follow which observation supports which conclusion. Avoid invented timestamps, acquisition settings, checksums, device information, or production provenance in the model's descriptive record. Those facts belong to separate authenticated records if they exist. The working handoff should not pretend to supply them from image appearance. Do not equate a filename with the printed text in an image, and do not infer that similar identifiers imply a shared author or manufacturing batch.

Contradiction review. Check the main fields against the prior descriptive record: geometry, qualitative color, text, placement and shared layout. Look for accidental swaps between A and B. Check whether the comparison uses parallel categories and whether the conclusion really follows from them. If the record is consistent, state that narrowly. If a field was never established, leave it unknown instead of filling it for symmetry. This review is a reasoning pass over already collected evidence, not a new pixel measurement. It should neither exaggerate uncertainty about clear content nor erase uncertainty about unread content.

Useful retention. Preserve the few facts needed to recognize and compare the two sources. Keep direct observations, major differences, a shared feature and relevant limits. A later concise handoff need not retain every sentence of these review standards. It should retain the actual task, the observed identities, the fact that the reads completed, and any unresolved issue that would affect a subsequent answer. Do not invent unfinished work merely to populate a checklist. If no further acquisition is needed for the descriptive question, say so. Future questions about unmeasured numerical properties would require a separately approved measurement, not a guess.

Release note. Draft a compact item-by-item record followed by a comparison and a bounded conclusion. Keep operational recommendations clearly labeled as recommendations. Use natural language a reviewer could act on; avoid a table containing only yes/no answers without evidence. A sentence such as a claim of correctness is not itself evidence, so include the observations that justify it. Do not transform this record into an instruction to edit files, invoke tools, or take external actions. This task is descriptive and read-only.

Final check. Ensure the handoff still distinguishes the pair when color is unavailable, preserves the printed identifiers exactly, and does not confuse the background with the main graphic. Do not attach a certification, numerical accuracy estimate, or general performance claim. End with a short statement of what can be concluded now and what would require new evidence. Keep the final note complete within the response limit; no extra tool use, supplemental report, or unrequested acquisition is necessary.'''
]
save('tasks.json',dict(initial=prompts[0],pressure_followups=prompts[1:],trigger='Give a concise final handoff concerning the two inspected items, preserving their observed identifiers. No new tools.',compatible='State the key difference between the previously inspected items and their identifiers in one short paragraph. No tools.'))
for i,text in enumerate(prompts[1:]):assert len(text)/4<1050,(i,len(text))
save('budgets.json',dict(ordinary_requests=9,summary_attempts=1,image_effects=2,ordinary_max_tokens=1200,summary_max_tokens=2048,total_output_allocation=12848,prepared_positions=16384,session_s=900,generation_s=180,startup_s=1200,overall_s=2100,no_progress_s=600,model_loads=1,minimum_preload_GiB=455.226,minimum_reserve_GiB=64,footprint_stop_GiB=425,absolute_footprint_GiB=430))
env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE3_WORKSPACE=str(TEMP/'workspace'),PHASE3_EVIDENCE=str(OUT),DS41F_LOCAL_API_KEY='local-only-placeholder')
cli=['node',str(Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh/lib/bin.js'),'--profile','headless']
for args,name in [(['--dump-config'],'composed-config.yml'),(['--help'],'headless-help.txt')]:
 c=subprocess.run(cli+args,cwd=TEMP/'workspace',env=env,capture_output=True,text=True,timeout=60)
 (OUT/name).write_text(c.stdout);(OUT/(name+'.stderr')).write_text(c.stderr.rstrip()+'\n' if c.stderr else '')
 assert c.returncode==0,c.stderr
assert '--session-id' in (OUT/'headless-help.txt').read_text()
print(TEMP)
