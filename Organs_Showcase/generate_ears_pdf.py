import math
from pathlib import Path
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def draw_arrow_head(c, x1, y1, x2, y2, color, size=6):
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux = dx / length
    uy = dy / length
    vx = -uy
    vy = ux

    p1 = (x2, y2)
    p2 = (x2 - size * ux + (size / 2.2) * vx, y2 - size * uy + (size / 2.2) * vy)
    p3 = (x2 - size * ux - (size / 2.2) * vx, y2 - size * uy - (size / 2.2) * vy)

    c.setFillColor(color)
    c.setStrokeColor(color)
    p = c.beginPath()
    p.moveTo(*p1)
    p.lineTo(*p2)
    p.lineTo(*p3)
    p.close()
    c.drawPath(p, fill=1, stroke=1)


def draw_polyline_arrow(c, points, color, label="", label_offset=(0, 0)):
    c.setStrokeColor(color)
    c.setLineWidth(1.3)
    for i in range(len(points) - 1):
        c.line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

    draw_arrow_head(c, points[-2][0], points[-2][1], points[-1][0], points[-1][1], color, size=7)

    if label:
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(color)
        lx = (points[0][0] + points[1][0]) / 2 + label_offset[0]
        ly = (points[0][1] + points[1][1]) / 2 + label_offset[1]
        c.drawString(lx, ly, label)


def draw_node_box(c, cx, cy, w, h, lines, fill_hex="#F8FAFC", stroke_hex="#475569", header_mode=False):
    x = cx - w / 2
    y = cy - h / 2
    c.setFillColor(colors.HexColor(fill_hex))
    c.setStrokeColor(colors.HexColor(stroke_hex))
    c.setLineWidth(1.4 if not header_mode else 2.0)
    c.roundRect(x, y, w, h, radius=5, fill=1, stroke=1)

    line_height = 11.5
    total_text_h = len(lines) * line_height
    start_y = cy + (total_text_h / 2) - 8.5

    for i, line in enumerate(lines):
        if header_mode:
            c.setFillColor(colors.HexColor("#FFFFFF"))
            c.setFont("Helvetica-Bold", 9)
        elif i == 0 and len(lines) > 1:
            c.setFillColor(colors.HexColor("#0F172A"))
            c.setFont("Helvetica-Bold", 8.5)
        else:
            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica", 8)
        c.drawCentredString(cx, start_y - (i * line_height), line)


def draw_diamond_node(c, cx, cy, w, h, lines, fill_hex="#FEF3C7", stroke_hex="#D97706"):
    p = c.beginPath()
    p.moveTo(cx, cy + h / 2)
    p.lineTo(cx + w / 2, cy)
    p.lineTo(cx, cy - h / 2)
    p.lineTo(cx - w / 2, cy)
    p.close()

    c.setFillColor(colors.HexColor(fill_hex))
    c.setStrokeColor(colors.HexColor(stroke_hex))
    c.setLineWidth(1.4)
    c.drawPath(p, fill=1, stroke=1)

    line_height = 10.5
    total_text_h = len(lines) * line_height
    start_y = cy + (total_text_h / 2) - 8

    for i, line in enumerate(lines):
        c.setFillColor(colors.HexColor("#92400E"))
        c.setFont("Helvetica-Bold", 7.8)
        c.drawCentredString(cx, start_y - (i * line_height), line)


def generate_pdf(output_path: Path):
    PAGE_WIDTH = 1200
    PAGE_HEIGHT = 1860
    c = canvas.Canvas(str(output_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    # Canvas Background
    c.setFillColor(colors.HexColor("#FAFAFA"))
    c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)

    # -------------------------------------------------------------------------
    # Header Banner
    # -------------------------------------------------------------------------
    c.setFillColor(colors.HexColor("#0F172A"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(PAGE_WIDTH / 2, 1815, "ATLAS VOICE ASSISTANT — EARS (ears.py) ARCHITECTURE FLOWCHART")

    c.setFillColor(colors.HexColor("#64748B"))
    c.setFont("Helvetica", 10.5)
    c.drawCentredString(PAGE_WIDTH / 2, 1795, "Asynchronous Ingestion | CPU Wake Spotting | Barge-In Interception | VAD Energy Bounding | GPU Transcription")

    # Legend Container
    c.setFillColor(colors.HexColor("#F1F5F9"))
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.setLineWidth(1)
    c.roundRect(100, 1740, 1000, 36, radius=4, fill=1, stroke=1)

    legend_items = [
        ("#DCFCE7", "#16A34A", "Lifecycle / Success"),
        ("#E0F2FE", "#0284C7", "Audio Streaming I/O"),
        ("#FEF3C7", "#D97706", "Decision Gates"),
        ("#FEE2E2", "#DC2626", "Barge-In / Termination"),
        ("#E0E7FF", "#4338CA", "Buffer & Data Relay"),
    ]
    lx = 130
    for f_col, s_col, text in legend_items:
        c.setFillColor(colors.HexColor(f_col))
        c.setStrokeColor(colors.HexColor(s_col))
        c.roundRect(lx, 1752, 14, 12, radius=2, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#334155"))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(lx + 20, 1754, text)
        lx += 195

    # -------------------------------------------------------------------------
    # Subgraph: Asynchronous Audio Thread
    # -------------------------------------------------------------------------
    c.setFillColor(colors.HexColor("#F8FAFC"))
    c.setStrokeColor(colors.HexColor("#94A3B8"))
    c.setLineWidth(1)
    c.roundRect(40, 1370, 260, 350, radius=6, fill=1, stroke=1)
    c.setFillColor(colors.HexColor("#475569"))
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(170, 1698, "ASYNCHRONOUS AUDIO THREAD")

    draw_node_box(c, 170, 1640, 220, 44, ["Microphone Input Stream", "sounddevice.InputStream(16kHz mono)"], "#E0F2FE", "#0284C7")
    draw_node_box(c, 170, 1530, 220, 44, ["_audio_callback(indata, frames)", "Capture 1,280 samples (80ms int16)"], "#E0F2FE", "#0284C7")
    draw_node_box(c, 170, 1420, 220, 44, ["audio_queue (FIFO)", "Thread-Safe Memory Queue"], "#E0E7FF", "#4338CA")

    draw_polyline_arrow(c, [(170, 1618), (170, 1552)], colors.HexColor("#0284C7"))
    draw_polyline_arrow(c, [(170, 1508), (170, 1442)], colors.HexColor("#0284C7"), "indata.copy()", (6, 0))

    # -------------------------------------------------------------------------
    # Main Column Nodes & Lifecycle
    # -------------------------------------------------------------------------
    draw_node_box(c, 600, 1670, 250, 44, ["main.py Invokes ears.start()", "Lifecycle Ignition & Supervisor"], "#DCFCE7", "#16A34A")
    draw_node_box(c, 600, 1575, 290, 52, ["Initialize Hardware & Models", "OWW on CPU | Whisper large-v3-turbo on GPU", "Thresholds: Wake=0.6, RMS=450, Window=10s"], "#F8FAFC", "#475569")
    draw_node_box(c, 600, 1475, 260, 44, ["Open sd.InputStream Stream", "Rate=16kHz, Blocksize=1280, int16"], "#F8FAFC", "#475569")
    draw_node_box(c, 600, 1385, 240, 42, ["Set state = 'IDLE'", "is_running = True"], "#F8FAFC", "#475569")
    draw_node_box(c, 600, 1290, 260, 46, ["Fetch Frame from audio_queue", "Pulls 80ms PCM frame (1280 samples)"], "#E0E7FF", "#4338CA")

    # Queue connection to Main Loop
    draw_polyline_arrow(c, [(280, 1420), (430, 1420), (430, 1290), (470, 1290)], colors.HexColor("#4338CA"), "audio_queue.get()", (10, 5))

    draw_polyline_arrow(c, [(600, 1648), (600, 1601)], colors.HexColor("#16A34A"))
    draw_polyline_arrow(c, [(600, 1549), (600, 1497)], colors.HexColor("#475569"))
    draw_polyline_arrow(c, [(600, 1453), (600, 1406)], colors.HexColor("#475569"))
    draw_polyline_arrow(c, [(600, 1364), (600, 1313)], colors.HexColor("#475569"))

    # -------------------------------------------------------------------------
    # Barge-In Decision & Path (Right Column)
    # -------------------------------------------------------------------------
    draw_diamond_node(c, 600, 1185, 210, 68, ["Is mouth.is_speaking", "== True?"])
    draw_polyline_arrow(c, [(600, 1267), (600, 1219)], colors.HexColor("#475569"))

    draw_node_box(c, 970, 1185, 220, 46, ["oww_model.predict(frame)", "Monitor triggers during audio output"], "#FEF3C7", "#D97706")
    draw_polyline_arrow(c, [(705, 1185), (860, 1185)], colors.HexColor("#DC2626"), "YES", (15, 6))

    draw_diamond_node(c, 970, 1085, 180, 62, ["Wake score >= 0.6?", "(Barge Trigger)"])
    draw_polyline_arrow(c, [(970, 1162), (970, 1116)], colors.HexColor("#D97706"))

    draw_node_box(c, 970, 975, 240, 56, ["BARGE-IN ENGAGED!", "mouth.stop() | oww.reset()", "state = 'ACTIVE' | _flush_queue()"], "#FEE2E2", "#DC2626")
    draw_polyline_arrow(c, [(970, 1054), (970, 1003)], colors.HexColor("#DC2626"), "YES", (6, 0))

    # Barge Loopback to Fetch
    draw_polyline_arrow(c, [(970, 947), (970, 910), (1110, 910), (1110, 1290), (730, 1290)], colors.HexColor("#DC2626"), "Interrupted -> Listen", (-60, -8))

    # -------------------------------------------------------------------------
    # State Router Node
    # -------------------------------------------------------------------------
    draw_diamond_node(c, 600, 1070, 180, 62, ["Current State?"])
    draw_polyline_arrow(c, [(600, 1151), (600, 1101)], colors.HexColor("#475569"), "NO", (6, 0))
    draw_polyline_arrow(c, [(880, 1085), (690, 1070)], colors.HexColor("#64748B"), "NO", (10, 6))

    # -------------------------------------------------------------------------
    # IDLE Branch (Left Column)
    # -------------------------------------------------------------------------
    draw_node_box(c, 300, 990, 220, 46, ["oww_model.predict(frame)", "Evaluate 'Atlas', 'Buddy', 'Pal'"], "#FEF3C7", "#D97706")
    draw_polyline_arrow(c, [(510, 1070), (300, 1070), (300, 1013)], colors.HexColor("#D97706"), "state == 'IDLE'", (30, 6))

    draw_diamond_node(c, 300, 885, 190, 64, ["Wake score >= 0.6?", "(Trigger Verified)"])
    draw_polyline_arrow(c, [(300, 967), (300, 917)], colors.HexColor("#D97706"))

    # Wake rejected loop back to fetch
    draw_polyline_arrow(c, [(205, 885), (100, 885), (100, 1290), (470, 1290)], colors.HexColor("#64748B"), "NO (Discard)", (-40, 8))

    draw_node_box(c, 300, 770, 240, 56, ["Engage Active Session", "oww.reset() | _flush_queue()", "state = 'ACTIVE' | last_active = now", "Optional: mouth.speak(wake_receipt)"], "#DCFCE7", "#16A34A")
    draw_polyline_arrow(c, [(300, 853), (300, 798)], colors.HexColor("#16A34A"), "YES", (6, 0))

    # Engage loopback to fetch
    draw_polyline_arrow(c, [(420, 770), (490, 770), (490, 1280), (470, 1280)], colors.HexColor("#16A34A"), "Next Turn", (15, 0))

    # -------------------------------------------------------------------------
    # ACTIVE Branch & Conversational Window Check
    # -------------------------------------------------------------------------
    draw_diamond_node(c, 600, 960, 220, 66, ["Elapsed > 10.0s?", "(Attention Window Expired)"])
    draw_polyline_arrow(c, [(600, 1039), (600, 993)], colors.HexColor("#475569"), "state == 'ACTIVE'", (6, 0))

    draw_node_box(c, 840, 885, 210, 46, ["Window Elapsed -> Revert", "oww.reset() | _flush_queue()", "state = 'IDLE'"], "#F1F5F9", "#64748B")
    draw_polyline_arrow(c, [(710, 960), (840, 960), (840, 908)], colors.HexColor("#64748B"), "YES", (15, 6))
    draw_polyline_arrow(c, [(840, 862), (840, 830), (1050, 830), (1050, 1280), (730, 1280)], colors.HexColor("#64748B"))

    # -------------------------------------------------------------------------
    # Utterance Capture & VAD Energy Loop
    # -------------------------------------------------------------------------
    draw_node_box(c, 600, 860, 260, 46, ["Initialize Utterance Buffer", "chunks=[frame], speech=False", "silence_frames=0, waiting_frames=0"], "#E0F2FE", "#0284C7")
    draw_polyline_arrow(c, [(600, 927), (600, 883)], colors.HexColor("#0284C7"), "NO (Active)", (6, 0))

    draw_node_box(c, 600, 770, 260, 46, ["Fetch Next Frame from audio_queue", "Append to chunks | Calc RMS Energy"], "#E0E7FF", "#4338CA")
    draw_polyline_arrow(c, [(600, 837), (600, 793)], colors.HexColor("#4338CA"))

    draw_diamond_node(c, 600, 675, 200, 66, ["RMS > 450.0?", "(Voice Boundary Check)"])
    draw_polyline_arrow(c, [(600, 747), (600, 708)], colors.HexColor("#4338CA"))

    draw_node_box(c, 410, 595, 200, 42, ["speech_detected = True", "silence_frames = 0"], "#DCFCE7", "#16A34A")
    draw_node_box(c, 790, 595, 230, 42, ["If speech: silence_frames += 1", "Else: waiting_frames += 1"], "#FEF3C7", "#D97706")

    draw_polyline_arrow(c, [(500, 675), (410, 675), (410, 616)], colors.HexColor("#16A34A"), "YES", (-15, 6))
    draw_polyline_arrow(c, [(700, 675), (790, 675), (790, 616)], colors.HexColor("#D97706"), "NO", (15, 6))

    draw_diamond_node(c, 600, 505, 240, 70, ["Cutoff Condition Met?", "Silence>=1.4s | Wait>=2.5s | Max>=12s"])
    draw_polyline_arrow(c, [(410, 574), (410, 535), (500, 505)], colors.HexColor("#16A34A"))
    draw_polyline_arrow(c, [(790, 574), (790, 535), (700, 505)], colors.HexColor("#D97706"))

    # VAD Loopback to next frame
    draw_polyline_arrow(c, [(480, 505), (280, 505), (280, 770), (470, 770)], colors.HexColor("#64748B"), "NO (Continue Record)", (-80, 6))

    # -------------------------------------------------------------------------
    # Transcription & Downstream Handoff
    # -------------------------------------------------------------------------
    draw_diamond_node(c, 600, 395, 220, 62, ["speech_detected == True?", "(Filter Ghost Triggers)"])
    draw_polyline_arrow(c, [(600, 470), (600, 426)], colors.HexColor("#475569"), "YES (Cutoff)", (6, 0))

    # Ghost trigger bailout
    draw_polyline_arrow(c, [(490, 395), (240, 395), (240, 1270), (470, 1270)], colors.HexColor("#EF4444"), "NO (Empty Audio)", (-60, 6))

    draw_node_box(c, 600, 295, 330, 54, ["take_input(recorded_chunks)", "Concat int16 -> Normalize float32 [-1, 1]", "faster-whisper GPU (large-v3-turbo, beam_size=1)"], "#E0F2FE", "#0284C7")
    draw_polyline_arrow(c, [(600, 364), (600, 322)], colors.HexColor("#0284C7"), "YES", (6, 0))

    # Command Classification
    draw_diamond_node(c, 600, 195, 220, 64, ["Command Reflex Check?", "Shutdown / Dismiss / Query"])
    draw_polyline_arrow(c, [(600, 268), (600, 227)], colors.HexColor("#475569"))

    draw_node_box(c, 240, 195, 230, 48, ["Shutdown Reflex", "mouth.speak('Deactivating...')", "ears.stop() -> SystemExit(0)"], "#FEE2E2", "#DC2626")
    draw_node_box(c, 960, 195, 230, 48, ["Dismissal Reflex", "mouth.speak(dismiss_receipt)", "state = 'IDLE' | _flush_queue()"], "#FEF3C7", "#D97706")

    draw_polyline_arrow(c, [(490, 195), (355, 195)], colors.HexColor("#DC2626"), "Shutdown", (20, 6))
    draw_polyline_arrow(c, [(710, 195), (845, 195)], colors.HexColor("#D97706"), "Dismissal", (20, 6))

    # Dismiss loopback to fetch
    draw_polyline_arrow(c, [(1075, 195), (1140, 195), (1140, 1270), (730, 1270)], colors.HexColor("#D97706"))

    # Brain / Cognitive Handoff
    draw_node_box(c, 600, 100, 360, 52, ["Downstream Cognitive Handoff", "brain.think(prompt) -> llm.py 5-Tier Waterfall", "mouth.speak(assistant_reply)"], "#DCFCE7", "#16A34A")
    draw_polyline_arrow(c, [(600, 163), (600, 126)], colors.HexColor("#16A34A"), "Standard Utterance", (6, 0))

    # Loopback after response finishes
    draw_node_box(c, 600, 22, 330, 32, ["Reset active timer | _flush_queue() -> Next Interaction Turn"], "#F1F5F9", "#475569")
    draw_polyline_arrow(c, [(600, 74), (600, 38)], colors.HexColor("#475569"))
    draw_polyline_arrow(c, [(765, 22), (1160, 22), (1160, 1260), (730, 1260)], colors.HexColor("#16A34A"))

    # Save to disk
    c.showPage()
    c.save()
    print(f"[Success] Ears flowchart PDF saved at: {output_path.resolve()}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    current_dir.mkdir(parents=True, exist_ok=True)
    pdf_destination = current_dir / "ears_architecture_flowchart.pdf"
    generate_pdf(pdf_destination)