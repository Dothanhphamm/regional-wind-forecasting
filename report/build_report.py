"""Build the editable first report from the archived Tufts scores and plot."""
import csv
import subprocess
from copy import deepcopy
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "wind-6afo1x"
ARMS = ("lstm", "plain", "both")
LABELS = ["Conventional LSTM", "PredRNN", "PredRNN + both constraints"]
NAMES = ["Conventional LSTM", "PredRNN without physical constraints",
         "PredRNN with mass and dry-static-energy constraints"]

def read(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))

summary = {(r["arm"], int(r["lead_h"])): r for r in read("summary.csv")}
history = {(int(r["training_years"]), int(r["lead_h"]), r["Model"]): r
           for r in read("validation_by_history.csv")}
gains = {int(r["lead_h"]): r for r in read("primary_gain.csv")}
seeds = {(r["arm"], int(r["seed"]), int(r["lead_h"])): float(r["wind_rmse_ms"])
         for r in read("per_seed.csv")}

def score(arm, lead, key="wind_rmse_ms"):
    return float(summary[arm, lead][key])

def gain_text(lead):
    row = gains[lead]
    gain, lo, hi = (float(row[k]) for k in ("wind_rmse_gain_ms", "lower95", "upper95"))
    return f"{gain:.3f} m/s at {lead} hours (95% interval {lo:.3f} to {hi:.3f})"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(.77), Inches(.70)
sec.left_margin, sec.right_margin = Inches(.90), Inches(.90)
sec.header_distance, sec.footer_distance = Inches(.32), Inches(.32)
ink = "18282E"
accent = "245F6A"
muted = "43585E"

def font(run, face="Arial", size=10.5, color=ink, bold=False, italic=False):
    run.font.name = face
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.bold, run.italic = bold, italic
    return run

normal = doc.styles["Normal"]
normal.font.name, normal.font.size = "Arial", Pt(10.5)
normal.font.color.rgb = RGBColor.from_string(ink)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.17

def p(txt, after=6):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(after)
    para.add_run(txt)
    return para

def heading(txt, level=1):
    para = doc.add_paragraph(style=f"Heading {level}")
    para.paragraph_format.space_before = Pt(0 if level == 1 else 8)
    para.paragraph_format.space_after = Pt(7 if level == 1 else 3)
    para.paragraph_format.keep_with_next = True
    font(para.add_run(txt), "Georgia" if level == 1 else "Arial",
         24 if level == 1 else 12, "18282E", bold=level != 1)
    return para

def kicker(txt):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(6)
    font(para.add_run(txt), size=9, bold=True, color=accent)

def note(txt):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after = Pt(5)
    font(para.add_run(txt), size=9, color=muted)

def table(headers, rows, widths, left=(0,)):
    t = doc.add_table(rows=0, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    for i, row in enumerate([headers, *rows]):
        cells = t.add_row().cells
        for j, (cell, label, width) in enumerate(zip(cells, row, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            prop = cell._tc.get_or_add_tcPr()
            if i == 0 or i % 2 == 0:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "DCEAED" if i == 0 else "F1F5F5")
                prop.append(shading)
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in left else WD_ALIGN_PARAGRAPH.RIGHT
            para.paragraph_format.space_before = Pt(3.5)
            para.paragraph_format.space_after = Pt(3.5)
            para.paragraph_format.line_spacing = 1.05
            font(para.add_run(str(label)), size=9.4, bold=i == 0)
    return t

def citation(label, url):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(2)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), doc.part.relate_to(url, RT.HYPERLINK, is_external=True))
    run = OxmlElement("w:r")
    prop = OxmlElement("w:rPr")
    for tag, val in (("color", accent), ("sz", "18"), ("u", "single")):
        el = OxmlElement("w:" + tag)
        el.set(qn("w:val"), val)
        prop.append(el)
    run.append(prop)
    node = OxmlElement("w:t")
    node.text = label
    run.append(node)
    link.append(run)
    para._p.append(link)

def add_mathematical_model():
    """Import the GitHub-readable specification, including editable Word equations."""
    converted = subprocess.run(
        ["pandoc", "--fail-if-warnings", "--from=markdown", "--to=docx", str(HERE / "mathematical-model.md")],
        check=True, stdout=subprocess.PIPE,
    )
    for source in Document(BytesIO(converted.stdout)).paragraphs:
        if source.style.name == "Heading 1":
            doc.add_page_break()
            kicker("MATHEMATICAL MODEL")
            heading(source.text)
        elif source.style.name == "Heading 2":
            heading(source.text, 2)
        else:
            target = doc.add_paragraph()
            if source._p.find(qn("m:oMathPara")) is not None:
                target.alignment = WD_ALIGN_PARAGRAPH.CENTER
                target.paragraph_format.space_before = Pt(1)
                target.paragraph_format.space_after = Pt(5)
            for child in source._p:
                if child.tag != qn("w:pPr"):
                    target._p.append(deepcopy(child))

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
font(footer.add_run("Regional wind forecasting   /   First report     •     "), size=8.5, color=muted)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)
doc.core_properties.title = "Do physical constraints improve regional wind forecasts?"
doc.core_properties.subject = "500 hPa wind forecasting using WeatherBench 2 ERA5"
doc.core_properties.author = ""
doc.core_properties.keywords = "WeatherBench 2, ERA5, PredRNN, 500 hPa winds"

kicker("FIRST RESEARCH REPORT     /     24 NOVEMBER 2026")
title = doc.add_paragraph(style="Title")
# A blank direct border overrides the default Title style's horizontal rule.
border = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
bottom.set(qn("w:val"), "nil")
border.append(bottom)
title._p.get_or_add_pPr().append(border)
title.paragraph_format.space_after = Pt(8)
title.paragraph_format.line_spacing = 1.04
font(title.add_run("Do physical constraints improve\nregional wind forecasts?"), "Georgia", 30, "151B1D")
p("Three recurrent models forecast 500 hPa winds over the Great Lakes and northeastern North America. We vary the amount of historical weather available for training, then ask whether two physical constraints improve a spatial model on a separate year.")
heading("What changed on the 2021 test", 2)
p("All three models learned from the same ten years. Switching from a conventional long short-term memory network (LSTM) to a PredRNN lowered 24-hour wind error by 0.99 m/s. Adding the mass and dry-static-energy constraints lowered it by another 0.10 m/s. At 48 hours, the corresponding reductions were 1.26 and 0.15 m/s.")
heading("Table 1. Mean 2021 vector wind RMSE", 2)
table(["Model", "24 h (m/s)", "48 h (m/s)"],
      [[name, *[f"{score(arm, h):.3f}" for h in (24, 48)]] for name, arm in zip(LABELS, ARMS)],
      [3.8, 1.45, 1.45])
note("RMSE means root mean squared error. Lower values indicate more accurate wind vectors. Each entry averages three separately trained models, scored on the same 362 daily forecasts.")
heading("Figure 1. Reduction in error from adding both constraints", 2)
para = doc.add_paragraph()
para.paragraph_format.space_after = Pt(0)
para.alignment = WD_ALIGN_PARAGRAPH.CENTER
para.add_run().add_picture(str(HERE / "physics-gain.png"), width=Inches(6.38))
note("Dots show plain PredRNN RMSE minus constrained PredRNN RMSE. Bars give pointwise 95% paired block-bootstrap intervals across 2021 dates. Teal marks the two leads used for model selection.")
p("The gain was " + gain_text(24) + " and " + gain_text(48) + ". That is about 1% of plain PredRNN error at each lead. The 48-hour improvement appeared in every seed.")

doc.add_page_break()
kicker("EXPERIMENT DESIGN")
heading("From regional weather maps to forecasts")
p("WeatherBench 2 provides a gridded version of ERA5, a reanalysis that combines observations with a numerical weather model [1, 2]. The regional subset contains modified Copernicus Climate Change Service information for 2008–2021. We use it both to initialize forecasts and to evaluate them. The target is the eastward and northward wind at 500 hPa, a pressure level in the middle troposphere.")
heading("One forecast case", 2)
p("Each day at 00 UTC, the model receives three maps from the previous 24 hours, spaced 12 hours apart. It predicts four successive maps at 12, 24, 36 and 48 hours, using its own previous forecast as input. Each case and all its targets lie within one time split.")
p("The six-hourly source has 5.625° spacing. The regional input contains 5 × 7 cells centered between 30.94° and 53.44° N and 95.63° and 61.88° W. One outer row or column on each side supports spatial derivatives, leaving 3 × 5 interior cells for scoring. The eleven forecast fields comprise two wind components at 500 hPa plus temperature, geopotential and pressure vertical velocity at 400, 500 and 600 hPa.")
heading("Table 2. Which weather each stage could use", 2)
table(["Stage", "Years", "Daily cases", "Role"], [
    ["Training", "2016–17", "728", "Shortest history"],
    ["Training", "2013–17", "1,823", "Intermediate history"],
    ["Training", "2008–17", "3,650", "Longest history"],
    ["Validation", "2018", "362", "Select checkpoints and weights"],
    ["Calibration", "2019", "362", "Fit wind prediction intervals"],
    ["Final test", "2021", "362", "Score the ten-year finalists"]], [1.22, 1.11, 1.14, 3.23], left=(0,1,3))
note("The three training windows are nested. Weather from 2020 served only for the earlier pilot and a workflow check. Normalization and physical penalty scales use each model’s own training window.")
heading("What differs among the three models", 2)
p("The conventional LSTM flattens each map and uses 24 hidden units. The regional PredRNN retains neighboring cells in two convolutional recurrent layers with eight hidden channels each. Both PredRNN versions share the architecture, memory-decoupling loss and training schedule [3].")
p("The third model adds two training penalties. The mass penalty compares horizontal wind divergence plus the vertical pressure derivative of vertical velocity with its ERA5 value. The energy penalty compares the change in dry static energy (1,004 × temperature + geopotential), including horizontal and vertical transport, with the corresponding ERA5 tendency. Both targets can be nonzero on this grid.")
note("Tufts used a CPU job array for 99 training trials. The two regional NetCDF files together occupy 34.45 MB. Most repeated computation comes from fitting models across training windows, weights and seeds.")
note("Neither the European Commission nor ECMWF is responsible for any use made of the Copernicus information in this report.")

add_mathematical_model()

doc.add_page_break()
kicker("MODEL SELECTION AND TRAINING HISTORY")
heading("How much did more weather help?")
p("On the same 362 dates in 2018, all three models had lower 24- and 48-hour errors as training expanded from two to five to ten years. At 24 hours, the constrained model fell from 10.97 to 8.84 m/s. Both PredRNN versions beat the conventional LSTM at each training length and lead shown.")
heading("Table 3. Vector wind RMSE on the common 2018 validation dates", 2)
hist_rows = []
for years in (2, 5, 10):
    for h in (24, 48):
        a, b, c = [float(history[years, h, name]["RMSE_m_s"]) for name in NAMES]
        hist_rows.append([f"{years} years", f"{h} h", f"{a:.3f}", f"{b:.3f}", f"{c:.3f}", f"{b-c:+.3f}"])
table(["History", "Lead", "LSTM", "PredRNN", "PredRNN\n+ both", "Physics\ngain"], hist_rows,
      [1.13, .76, 1.03, 1.22, 1.31, 1.25])
note("All values are m/s, averaged over seeds 42–44. Positive physics gain means a lower error after adding both penalties. Validation was also used to select model settings.")
heading("The gain from constraints depended on the training window", 2)
p("With 728 training cases, the constraints slightly worsened the 24-hour forecast and barely changed the 48-hour result. The largest validation gains occurred with five years. Ten years produced the lowest absolute errors, although the incremental gain from physics was smaller than with five years. The 2021 test evaluated only the ten-year models.")
heading("How the settings were selected", 2)
p("All models used Adam at a learning rate of 0.0001, batches of 32, up to 60 epochs and early stopping after ten epochs without improvement. Within each history, each seed retained the epoch with the lowest mean of 24- and 48-hour validation RMSE. The constrained configuration then minimized that score averaged over all three seeds.")
p("Each history included three conventional LSTM fits, three plain PredRNN fits and 27 constrained fits. The latter covered nine weight pairs across the same three seeds. The selected mass weights rose from 0.003 to 0.010 to 0.030 across the three histories. The energy weight was 0.003 each time.")
p("Longer histories also give the optimizer more batches per epoch. The validation pattern therefore measures what this training procedure achieved with more data. It cannot isolate the contribution of additional weather years from additional optimization steps.")

doc.add_page_break()
kicker("INTERPRETATION AND REPRODUCIBILITY")
heading("How strong is the evidence for a physics gain?")
p("The ten-year constrained model reduced RMSE by 1.09% at 24 hours and 1.25% at 48 hours. Forecast error weights grid cells by latitude and combines squared errors in both wind components before taking a square root. The comparison averages seed-specific RMSEs. Each model faces exactly the same weather.")
heading("Table 4. Physics gain by training seed on the 2021 test", 2)
table(["Lead", "Seed 42", "Seed 43", "Seed 44", "Mean"],
      [[f"{h} h", *[f'{seeds["plain", s, h]-seeds["both", s, h]:+.3f}' for s in (42,43,44)],
        f'{float(gains[h]["wind_rmse_gain_ms"]):+.3f}'] for h in (24,48)], [1.34]*5)
note("Gain is in m/s. The 24-hour mean was driven largely by seed 44. At 48 hours, every seed improved, with the largest gain again coming from seed 44.")
p("The intervals in Figure 1 resample consecutive seven-day blocks, wrapping at year-end, with the same dates for both models and all seeds in each of 1,000 draws. Both primary intervals exclude zero. They describe weather-date uncertainty within 2021, conditional on these fitted models. One region, one test year and three seeds limit claims about performance elsewhere.")
heading("What the other measures say", 2)
p("Relative to plain PredRNN, the constrained model reduced mass-diagnostic RMSE by 28.7% at 24 hours and 31.0% at 48 hours. Dry-static-energy tendency RMSE fell by 5.7% and 7.1%. The larger changes in the diagnostics are consistent with the loss directly rewarding agreement with ERA5. The joint comparison cannot assign the wind improvement to either constraint alone.")
p("Spatial similarity (SSIM) on wind-speed maps improved at both leads. Perceptual distance (LPIPS, lower is better) slightly worsened at 24 hours and rose from 0.1138 to 0.1187 at 48 hours. These image metrics discard wind direction, and LPIPS upsamples only 15 cells [4, 5]. Intervals calibrated to 90% in 2019 covered 90.0–90.9% of the constrained model’s 2021 wind components across the two leads.")
p("The result supports a modest improvement in upper-air forecast accuracy for this region and year. The experiment provides no decision threshold for economic value. A direct replication would freeze these settings and evaluate another year before making a stronger claim.")
heading("Sources and replication", 2)
citation("[1] WeatherBench 2. Data guide and regridded ERA5 archive.", "https://weatherbench2.readthedocs.io/en/latest/data-guide.html")
citation("[2] Hersbach et al. (2020). The ERA5 global reanalysis.", "https://doi.org/10.1002/qj.3803")
citation("[3] Y. Wang et al. PredRNN. Spatiotemporal predictive learning with recurrent networks.", "https://arxiv.org/abs/2103.09504")
citation("[4] Z. Wang et al. (2004). Image quality assessment using structural similarity.", "https://ece.uwaterloo.ca/~z70wang/publications/ssim.html")
citation("[5] Zhang et al. (2018). The unreasonable effectiveness of deep features as a perceptual metric.", "https://richzhang.github.io/PerceptualSimilarity/")
note("Release v1 includes the scoring code, regional weather, all 99 fitted checkpoints and the original CSVs from Tufts run wind-6afo1x. The release README gives the steps to reproduce the tables and figure.")
doc.save(HERE / "regional-wind-report.docx")
