const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.author = "Sreeshanth Sivanantham";
pres.title = "MSc Progress Meeting";

// ---- palette ----
const NAVY = "0F2A43";   // dark bg (title/closing)
const BLUE = "1C6A8F";   // primary
const TEAL = "2A9D8F";   // supporting
const ACCENT = "E76F51"; // sharp accent (ties to red violation boxes)
const PAGE = "F7F9FB";   // content bg
const CARD = "EEF3F7";   // card tint
const INK = "1B2733";    // body on light
const MUTE = "5A6B7B";   // captions
const WHITE = "FFFFFF";
const HEAD = "Cambria";
const BODY = "Calibri";

const IMG = "/home/claude/msc_surveillance/output";
const sh = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 90, opacity: 0.16 });

function header(slide, num, titleText) {
  slide.background = { color: PAGE };
  slide.addShape(pres.shapes.OVAL, { x: 0.6, y: 0.45, w: 0.55, h: 0.55, fill: { color: BLUE } });
  slide.addText(String(num), { x: 0.6, y: 0.45, w: 0.55, h: 0.55, align: "center", valign: "middle",
    fontFace: HEAD, fontSize: 20, bold: true, color: WHITE, margin: 0 });
  slide.addText(titleText, { x: 1.35, y: 0.45, w: 8.0, h: 0.55, align: "left", valign: "middle",
    fontFace: HEAD, fontSize: 25, bold: true, color: NAVY, margin: 0 });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, shadow: sh() });
}

// ============ SLIDE 1 — TITLE ============
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("MSc PROGRESS MEETING  ·  WEEK S2", { x: 0.6, y: 0.55, w: 6.0, h: 0.4,
    fontFace: BODY, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
  s.addText("Deep Learning Based Real-Time Face Mask Detection and Social Distance Analysis in Surveillance Environments",
    { x: 0.6, y: 1.05, w: 5.5, h: 2.5, fontFace: HEAD, fontSize: 25, bold: true, color: WHITE, lineSpacingMultiple: 1.02 });
  s.addText("Improving and rigorously evaluating a COVID-era surveillance pipeline.",
    { x: 0.62, y: 3.55, w: 5.4, h: 0.5, fontFace: BODY, fontSize: 12, italic: true, color: "AECBDD" });
  s.addText([
    { text: "Sreeshanth Sivanantham", options: { bold: true, fontSize: 16, color: WHITE, breakLine: true } },
    { text: "MSc Advanced Computer Science (Artificial Intelligence)", options: { fontSize: 12, color: "CADCEC", breakLine: true } },
    { text: "Supervisor: Sebastian Ordyniak", options: { fontSize: 12, color: "CADCEC", breakLine: true } },
    { text: "COMP5200M · University of Leeds · 30 June 2026", options: { fontSize: 11, color: "8FAAC0" } },
  ], { x: 0.6, y: 4.35, w: 5.6, h: 1.1, fontFace: BODY, lineSpacingMultiple: 1.05, margin: 0 });

  // prototype image on the right with frame
  card(s, 6.18, 1.55, 3.34, 2.05, WHITE);
  s.addImage({ path: `${IMG}/demo_frame.png`, x: 6.3, y: 1.66, w: 3.1, h: 1.74 });
  s.addText("Live prototype output (this project)", { x: 6.18, y: 3.62, w: 3.34, h: 0.3,
    align: "center", fontFace: BODY, fontSize: 10, italic: true, color: "AECBDD", margin: 0 });
  s.addNotes("Open by stating the title and that this is the Week S2 progress meeting. Make the one-line framing: the topic is fixed and dated, so the value is in improving and rigorously evaluating the baseline. Point at the prototype image: 'this already runs.'");
}

// ============ SLIDE 2 — AIM & RESEARCH QUESTION ============
{
  const s = pres.addSlide();
  header(s, 1, "Aim & Research Question");
  card(s, 0.6, 1.35, 4.3, 3.65, CARD);
  s.addText("AIM", { x: 0.9, y: 1.6, w: 3.7, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: TEAL, charSpacing: 1, margin: 0 });
  s.addText("Design, implement, and evaluate a real-time system that detects face-mask non-compliance and social-distancing violations from surveillance video, to support safer public environments.",
    { x: 0.9, y: 2.0, w: 3.7, h: 2.8, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.1, valign: "top" });

  card(s, 5.1, 1.35, 4.3, 3.65, NAVY);
  s.addText("RESEARCH QUESTION", { x: 5.4, y: 1.6, w: 3.7, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: ACCENT, charSpacing: 1, margin: 0 });
  s.addText("How well do COVID-era mask and distancing pipelines generalise to demographically diverse, real-world surveillance — and can perspective-corrected distancing and bias-aware data measurably improve their reliability?",
    { x: 5.4, y: 2.0, w: 3.7, h: 2.8, fontFace: HEAD, fontSize: 16, italic: true, color: WHITE, lineSpacingMultiple: 1.12, valign: "top" });
  s.addNotes("The aim is the system; the research question is what lifts it above a 2021 re-run. Stress 'generalise to diverse populations' and 'measurably improve' — everything in the project becomes evidence for or against this question.");
}

// ============ SLIDE 3 — BASELINE ============
{
  const s = pres.addSlide();
  header(s, 2, "The Starting Baseline");
  s.addText("Two open-source MIT projects, combined — my cited reference point, not my contribution.",
    { x: 0.6, y: 1.2, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });

  const labels = ["Video feed", "Person detection\n(YOLOv3)", "Face + mask\n(MobileNetV2)", "Distance\n(pixel threshold)", "Alerts / overlay"];
  const fills = [BLUE, BLUE, BLUE, ACCENT, BLUE];
  const bw = 1.55, gap = 0.36; let x = 0.6;
  labels.forEach((lab, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.3, w: bw, h: 1.15, rectRadius: 0.08, fill: { color: fills[i] }, shadow: sh() });
    s.addText(lab, { x, y: 2.3, w: bw, h: 1.15, align: "center", valign: "middle", fontFace: BODY, fontSize: 11.5, bold: true, color: WHITE, margin: 2 });
    if (i < labels.length - 1) s.addText("›", { x: x + bw - 0.02, y: 2.3, w: gap + 0.04, h: 1.15, align: "center", valign: "middle", fontFace: BODY, fontSize: 22, bold: true, color: MUTE, margin: 0 });
    x += bw + gap;
  });
  s.addText("The orange stage (pixel-threshold distance) is the baseline's weakest point — see next slide.",
    { x: 0.6, y: 3.75, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: INK, margin: 0 });
  s.addText("Sources: chandrikadeb7/Face-Mask-Detection · saimj7/Social-Distancing-Detection-in-Real-Time (both MIT, cited).",
    { x: 0.6, y: 4.75, w: 8.8, h: 0.35, fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0 });
  s.addNotes("Be upfront: this is two public repos stitched together and is my reference, fully cited. Walk left-to-right through the pipeline. Flag the orange box now so the next slide lands.");
}

// ============ SLIDE 4 — LIMITATIONS ============
{
  const s = pres.addSlide();
  header(s, 3, "Where the Baseline Falls Short");
  const items = [
    ["YOLOv3 detector", "Outdated — slower and less accurate than current detectors."],
    ["Pixel-threshold distance", "No camera calibration; wrong across perspective."],
    ["No person tracking", "Per-frame only; cannot follow a person over time."],
    ["One narrow dataset", "Mostly East-Asian faces; untested on others."],
    ["Privacy not built", "Discussed in the report, never implemented."],
  ];
  const cw = 2.86, ch = 1.62, gx = 0.2, gy = 0.18;
  const xs = [0.6, 0.6 + cw + gx, 0.6 + 2 * (cw + gx)];
  const ys = [1.35, 1.35 + ch + gy];
  items.forEach((it, i) => {
    const x = xs[i % 3], y = ys[Math.floor(i / 3)];
    card(s, x, y, cw, ch, WHITE);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.22, w: 0.42, h: 0.42, fill: { color: ACCENT } });
    s.addText(String(i + 1), { x: x + 0.2, y: y + 0.22, w: 0.42, h: 0.42, align: "center", valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, margin: 0 });
    s.addText(it[0], { x: x + 0.74, y: y + 0.22, w: cw - 0.9, h: 0.45, fontFace: BODY, fontSize: 14, bold: true, color: NAVY, valign: "middle", margin: 0 });
    s.addText(it[1], { x: x + 0.2, y: y + 0.74, w: cw - 0.4, h: 0.74, fontFace: BODY, fontSize: 11.5, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.04 });
  });
  // summary card (6th cell)
  const x = xs[2], y = ys[1];
  card(s, x, y, cw, ch, NAVY);
  s.addText("Each gap is a concrete opportunity for an MSc-level contribution.",
    { x: x + 0.22, y: y, w: cw - 0.44, h: ch, align: "left", valign: "middle", fontFace: HEAD, fontSize: 14.5, italic: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.08 });
  s.addNotes("This is the pivot. Don't apologise for the baseline's flaws — they are the project. Name each one quickly; the dark card reframes the whole list as opportunity.");
}

// ============ SLIDE 5 — CONTRIBUTIONS ============
{
  const s = pres.addSlide();
  header(s, 4, "My Contributions");
  const rows = [
    [{ text: "Baseline weakness", options: { bold: true, color: WHITE, fill: { color: BLUE }, fontSize: 13, align: "left" } },
     { text: "My contribution", options: { bold: true, color: WHITE, fill: { color: BLUE }, fontSize: 13, align: "left" } }],
    ["Outdated YOLOv3", "Modern tracked detector (YOLOv8/11 + ByteTrack)"],
    ["Pixel-threshold distance", "Perspective-corrected metric distance (homography → metres)"],
    ["No tracking", "Per-person tracking with violation history"],
    ["Narrow, biased dataset", "Bias-aware cross-dataset evaluation (BAFMD)"],
    ["Privacy not implemented", "Face anonymisation built + honest, rigorous evaluation"],
  ];
  const styled = rows.map((r, ri) => r.map((c, ci) => {
    if (ri === 0) return c;
    return { text: c, options: { fontSize: 12.5, color: INK, fill: { color: ri % 2 ? CARD : WHITE },
      bold: ci === 1, align: "left", valign: "middle" } };
  }));
  s.addTable(styled, { x: 0.6, y: 1.4, w: 8.8, colW: [3.4, 5.4], rowH: 0.55,
    border: { type: "solid", pt: 0.5, color: "D5DEE6" }, margin: [4, 6, 4, 6], valign: "middle", fontFace: BODY });
  s.addText("Same fixed topic and title — the contribution is depth, method, and rigour, not the subject.",
    { x: 0.6, y: 4.85, w: 8.8, h: 0.35, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, margin: 0 });
  s.addNotes("Read this as 'for every weakness, here's my fix.' The headline contributions are the perspective-corrected distance and the bias-aware evaluation. Reassure the assessor the title is unchanged.");
}

// ============ SLIDE 6 — DEMO / PROGRESS ============
{
  const s = pres.addSlide();
  header(s, 5, "Progress So Far: Working Prototype");
  s.addText([
    { text: "Re-implemented as a clean, modular, original pipeline — runs end-to-end.", options: { bullet: true, breakLine: true } },
    { text: "Person detection + multi-object tracking integrated.", options: { bullet: true, breakLine: true } },
    { text: "Perspective-corrected distance module built and unit-tested.", options: { bullet: true, breakLine: true } },
    { text: "Privacy face-blurring implemented.", options: { bullet: true, breakLine: true } },
    { text: "Runs on CPU today; real-time on GPU.", options: { bullet: true } },
  ], { x: 0.6, y: 1.5, w: 4.2, h: 3.2, fontFace: BODY, fontSize: 14, color: INK, paraSpaceAfter: 10, valign: "top" });

  card(s, 5.0, 1.45, 4.4, 2.72, WHITE);
  s.addImage({ path: `${IMG}/demo_frame.png`, x: 5.12, y: 1.57, w: 4.16, h: 2.34 });
  s.addText("Prototype output on test footage — tracked people, close-contact violations flagged in red.",
    { x: 5.0, y: 4.25, w: 4.4, h: 0.6, align: "center", fontFace: BODY, fontSize: 11, italic: true, color: MUTE, margin: 0, lineSpacingMultiple: 1.05 });
  s.addNotes("This is your proof you've moved past planning. Mention you ran it yourself on a Mac. The image is your own output, not the baseline's.");
}

// ============ SLIDE 7 — EVALUATION ============
{
  const s = pres.addSlide();
  header(s, 6, "Evaluation: Early Result & Plan");
  card(s, 0.6, 1.45, 2.85, 2.95, WHITE);
  s.addImage({ path: `${IMG}/mask_eval_confusion_matrix.png`, x: 0.78, y: 1.6, w: 2.5, h: 2.5 });
  s.addText("Mask classifier, held-out test (613 images).", { x: 0.6, y: 4.42, w: 2.85, h: 0.3, align: "center", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });

  s.addText("96%*", { x: 3.6, y: 1.7, w: 2.4, h: 0.95, align: "center", fontFace: HEAD, fontSize: 52, bold: true, color: ACCENT, margin: 0 });
  s.addText("in-dataset accuracy", { x: 3.6, y: 2.62, w: 2.4, h: 0.3, align: "center", fontFace: BODY, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  s.addText("*optimistic — tested on the model's own training data. Honest cross-dataset numbers are the real target.",
    { x: 3.6, y: 3.0, w: 2.4, h: 1.2, align: "center", fontFace: BODY, fontSize: 11, italic: true, color: MUTE, margin: 0, lineSpacingMultiple: 1.06 });

  card(s, 6.2, 1.45, 3.2, 3.4, CARD);
  s.addText("Evaluation plan", { x: 6.42, y: 1.62, w: 2.8, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: TEAL, margin: 0 });
  s.addText([
    { text: "Pixel vs perspective distance — controlled before/after.", options: { bullet: true, breakLine: true } },
    { text: "BAFMD cross-dataset generalisation gap.", options: { bullet: true, breakLine: true } },
    { text: "FPS measured on stated GPU hardware.", options: { bullet: true, breakLine: true } },
    { text: "Error analysis: lighting, crowd density, occlusion.", options: { bullet: true } },
  ], { x: 6.42, y: 2.05, w: 2.78, h: 2.6, fontFace: BODY, fontSize: 12, color: INK, paraSpaceAfter: 8, valign: "top" });
  s.addNotes("Lead with honesty: the 96% is flattering because it's the model's own data. That candour signals maturity. The plan slide shows you know what a real evaluation looks like.");
}

// ============ SLIDE 8 — ETHICS ============
{
  const s = pres.addSlide();
  header(s, 7, "Responsible Use & Ethics");
  const items = [
    ["Bias, quantified", "Measure performance gaps across demographics (BAFMD) and report them openly."],
    ["Privacy by design", "Face anonymisation implemented; minimise retained personal data."],
    ["Human in the loop", "A support tool for operators — not automated enforcement."],
    ["Transparency", "Limitations stated plainly; no overclaiming of accuracy."],
  ];
  const cw = 4.3, ch = 1.6, gx = 0.2, gy = 0.18;
  const xs = [0.6, 0.6 + cw + gx], ys = [1.4, 1.4 + ch + gy];
  items.forEach((it, i) => {
    const x = xs[i % 2], y = ys[Math.floor(i / 2)];
    card(s, x, y, cw, ch, WHITE);
    s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.3, w: 0.5, h: 0.5, fill: { color: TEAL } });
    s.addText(String.fromCharCode(65 + i), { x: x + 0.22, y: y + 0.3, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: HEAD, fontSize: 18, bold: true, color: WHITE, margin: 0 });
    s.addText(it[0], { x: x + 0.86, y: y + 0.28, w: cw - 1.1, h: 0.45, fontFace: BODY, fontSize: 15, bold: true, color: NAVY, valign: "middle", margin: 0 });
    s.addText(it[1], { x: x + 0.86, y: y + 0.74, w: cw - 1.1, h: 0.7, fontFace: BODY, fontSize: 12, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.05 });
  });
  s.addText("Ethics is integrated into the results, not a tacked-on chapter — and it is weighted at Leeds.",
    { x: 0.6, y: 4.75, w: 8.8, h: 0.35, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, margin: 0 });
  s.addNotes("Tie bias to your evaluation results, and point out the privacy blurring is actually built. This is where your project is unusually strong — lean in.");
}

// ============ SLIDE 9 — PLAN & RISKS ============
{
  const s = pres.addSlide();
  header(s, 8, "Plan to Submission (31 Aug) & Risks");
  const steps = [
    ["S3", "Refine design after this meeting"],
    ["S4–S5", "Testing & validation; draft Chapter 4"],
    ["S6", "Finalise Chapter 4; write Chapter 5"],
    ["S7", "Conclusions; finalise dissertation"],
    ["31 Aug", "Submit"],
  ];
  let y = 1.45;
  steps.forEach((st, i) => {
    s.addShape(pres.shapes.OVAL, { x: 0.6, y: y, w: 0.62, h: 0.62, fill: { color: i === 4 ? ACCENT : BLUE } });
    s.addText(st[0], { x: 0.58, y: y, w: 0.66, h: 0.62, align: "center", valign: "middle", fontFace: BODY, fontSize: st[0].length > 3 ? 10 : 13, bold: true, color: WHITE, margin: 0 });
    s.addText(st[1], { x: 1.4, y: y, w: 3.6, h: 0.62, align: "left", valign: "middle", fontFace: BODY, fontSize: 13, color: INK, margin: 0 });
    y += 0.72;
  });

  card(s, 5.4, 1.45, 4.0, 3.5, CARD);
  s.addText("Key risks & mitigations", { x: 5.62, y: 1.62, w: 3.6, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: TEAL, margin: 0 });
  s.addText([
    { text: "Dataset approval delay / dead Twitter links — request already sent; diverse fallback ready.", options: { bullet: true, breakLine: true } },
    { text: "GPU access for real-time — measure FPS on Leeds machines; CPU fallback works.", options: { bullet: true, breakLine: true } },
    { text: "Calibration accuracy — use a known-size reference; report distance as approximate.", options: { bullet: true } },
  ], { x: 5.62, y: 2.05, w: 3.58, h: 2.8, fontFace: BODY, fontSize: 12, color: INK, paraSpaceAfter: 10, valign: "top" });
  s.addNotes("Acknowledge the timeline is tight and you're slightly behind the plan — normal, and the meeting is formative. Showing you've named the risks and have mitigations is what reassures the assessor.");
}

// ============ SLIDE 10 — CLOSING ============
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.OVAL, { x: 8.7, y: -0.7, w: 2.4, h: 2.4, fill: { color: BLUE, transparency: 70 } });
  s.addShape(pres.shapes.OVAL, { x: -0.6, y: 4.2, w: 2.2, h: 2.2, fill: { color: TEAL, transparency: 75 } });
  s.addText("Questions & Discussion", { x: 0.7, y: 1.0, w: 8.6, h: 0.9, fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, margin: 0 });
  s.addText("My questions for you:", { x: 0.72, y: 2.05, w: 8.6, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: ACCENT, margin: 0 });
  s.addText([
    { text: "Is the scope and contribution pitched right for an MSc distinction?", options: { bullet: true, breakLine: true } },
    { text: "What would a strong mark look like for this project?", options: { bullet: true } },
  ], { x: 0.72, y: 2.5, w: 8.4, h: 1.3, fontFace: BODY, fontSize: 16, color: "DCE7F0", paraSpaceAfter: 10 });
  s.addText("Sreeshanth Sivanantham · COMP5200M · University of Leeds · 30 June 2026",
    { x: 0.7, y: 4.95, w: 8.6, h: 0.3, fontFace: BODY, fontSize: 11, color: "8FAAC0", margin: 0 });
  s.addNotes("Close by inviting feedback and asking your two questions directly. The assessor's answer on what a strong mark looks like is the most valuable thing you can leave with.");
}

pres.writeFile({ fileName: "/home/claude/msc_surveillance/Progress_Presentation.pptx" })
  .then(() => console.log("DECK WRITTEN"));
