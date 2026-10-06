import re
import math
from itertools import combinations

import pandas as pd
import streamlit as st

# PDF
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


# =========================================================
# Konfigurasi halaman
# =========================================================
st.set_page_config(
    page_title="Cosine Similarity — Data Mining",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
        .main { background: #f8fafc; }
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
            border: 1px solid #e2e8f0;
            margin-bottom: 1.2rem;
        }
        .hero h1 { margin: 0; color: #172554; font-size: 2.1rem; }
        .hero p { margin: .45rem 0 0; color: #475569; }
        .formula {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            font-size: 1.05rem;
            color: #1e293b;
            margin: .8rem 0 1rem;
        }
        .small-note { color: #64748b; font-size: .9rem; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: .8rem;
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Data default dari soal pada PDF
# =========================================================
DEFAULT_SENTENCES = {
    "S1": "Pulang kerja mampir kemana dulu hari ini?",
    "S2": "Hari ini pulang kerja cepat, main kemana ya?",
    "S3": "Besok jalan yuk, kemana sih pulang kerja.",
    "S4": "Nanti kemana ya pulang kerja.",
    "S5": "Pengen main, kemana ya pulang kerja ntar.",
}


# =========================================================
# Fungsi input & PDF
# =========================================================
def normalize_lines(text: str) -> list[str]:
    """Membersihkan teks hasil input/OCR tanpa mengubah isi kalimat."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return [line for line in lines if line]


def parse_pdf(pdf_file) -> tuple[dict[str, str], str]:
    """
    Membaca PDF.
    1) Coba ambil text layer.
    2) Jika PDF berupa scan/gambar, gunakan OCR Tesseract.
    Kalimat hasil OCR ditampilkan agar user dapat mengecek sebelum dihitung.
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    all_text = []

    for page in doc:
        text = page.get_text("text").strip()

        if text:
            all_text.append(text)
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            try:
                # Bahasa Indonesia jika tersedia; fallback ke English.
                try:
                    ocr_text = pytesseract.image_to_string(img, lang="ind")
                except Exception:
                    ocr_text = pytesseract.image_to_string(img, lang="eng")
                all_text.append(ocr_text)
            except Exception as exc:
                all_text.append("")
                st.warning(
                    "OCR PDF tidak dapat dijalankan. "
                    "Pastikan Tesseract OCR tersedia pada komputer/server."
                )

    raw_text = "\n".join(all_text).strip()
    lines = normalize_lines(raw_text)

    # Prioritaskan baris yang terlihat seperti S1/S2/... jika ada.
    labeled = {}
    for line in lines:
        match = re.match(r"^\s*(S\d+)\s*[:.)-]\s*(.+)$", line, flags=re.I)
        if match:
            labeled[match.group(1).upper()] = match.group(2).strip()

    if len(labeled) >= 2:
        sentences = dict(sorted(labeled.items(), key=lambda x: int(x[0][1:])))
    else:
        # Ambil baris sebagai kalimat, maksimal 20.
        candidates = [
            line for line in lines
            if len(re.findall(r"[A-Za-zÀ-ÿ]+", line)) >= 2
        ][:20]
        sentences = {f"S{i}": line for i, line in enumerate(candidates, start=1)}

    return sentences, raw_text


# =========================================================
# Fungsi perhitungan — TIDAK mengubah rumus
# =========================================================
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())


def build_binary_matrix(sentences: dict[str, str]):
    token_sets = {sid: set(tokenize(text)) for sid, text in sentences.items()}
    vocabulary = sorted(set().union(*token_sets.values())) if token_sets else []

    matrix = pd.DataFrame(
        {
            sid: [1 if word in token_sets[sid] else 0 for word in vocabulary]
            for sid in sentences
        },
        index=vocabulary,
    )
    return matrix, token_sets


def cosine_from_binary(matrix: pd.DataFrame, s1: str, s2: str):
    x = matrix[s1].to_numpy()
    y = matrix[s2].to_numpy()

    dot_product = int(x @ y)
    norm_x = math.sqrt(int(x @ x))
    norm_y = math.sqrt(int(y @ y))

    if norm_x == 0 or norm_y == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm_x * norm_y)

    common_words = matrix.index[
        (matrix[s1] == 1) & (matrix[s2] == 1)
    ].tolist()

    return {
        "s1": s1,
        "s2": s2,
        "common_words": common_words,
        "dot_product": dot_product,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "similarity": similarity,
    }


def calculate_all_pairs(matrix: pd.DataFrame):
    results = []
    for s1, s2 in combinations(matrix.columns, 2):
        results.append(cosine_from_binary(matrix, s1, s2))
    return pd.DataFrame(results)


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>📐 Cosine Similarity — Data Mining</h1>
        <p>
            Sistem perhitungan kemiripan antar kalimat menggunakan
            <b>Cosine Similarity</b> dengan representasi kata biner (0/1).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="formula">
        <b>Rumus:</b>
        &nbsp; Cos(X,Y) = X · Y / (|X| × |Y|)
        <br>
        <span class="small-note">
            X · Y = jumlah kata yang sama &nbsp;|&nbsp;
            |X| = √(jumlah kata pada X) &nbsp;|&nbsp;
            |Y| = √(jumlah kata pada Y)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar input
# =========================================================
with st.sidebar:
    st.header("⚙️ Sumber Data")

    input_mode = st.radio(
        "Pilih sumber input",
        ["Input Manual", "Upload PDF", "Data Contoh PDF"],
        index=0,
    )

    sentences = {}
    pdf_raw_text = ""

    if input_mode == "Data Contoh PDF":
        sentences = DEFAULT_SENTENCES.copy()

    elif input_mode == "Input Manual":
        st.caption("Masukkan minimal 2 kalimat.")
        n = st.number_input(
            "Jumlah kalimat",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
        )

        for i in range(1, int(n) + 1):
            sentences[f"S{i}"] = st.text_area(
                f"Kalimat S{i}",
                value="",
                height=80,
                placeholder=f"Tulis kalimat S{i}...",
            )

    elif input_mode == "Upload PDF":
        uploaded_pdf = st.file_uploader(
            "Upload file PDF",
            type=["pdf"],
            help="PDF dapat berupa PDF teks atau PDF hasil scan/gambar.",
        )

        if uploaded_pdf is not None:
            with st.spinner("Membaca PDF..."):
                sentences, pdf_raw_text = parse_pdf(uploaded_pdf)

            if sentences:
                st.success(f"{len(sentences)} kalimat berhasil dibaca dari PDF.")
            else:
                st.warning(
                    "Kalimat belum berhasil dikenali dari PDF. "
                    "Coba gunakan PDF yang berisi teks atau periksa hasil OCR."
                )

    st.divider()
    decimals = st.slider("Jumlah angka desimal", 2, 6, 3)

    st.caption(
        "Perhitungan tetap menggunakan representasi biner 0/1 "
        "dan rumus Cosine Similarity pada tugas."
    )

# =========================================================
# Review hasil PDF sebelum dihitung
# =========================================================
if input_mode == "Upload PDF" and pdf_raw_text:
    with st.expander("🔎 Lihat teks hasil pembacaan PDF", expanded=False):
        st.text(pdf_raw_text[:12000])

# =========================================================
# Validasi
# =========================================================
valid_sentences = {
    sid: text.strip()
    for sid, text in sentences.items()
    if isinstance(text, str) and text.strip()
}

if len(valid_sentences) < 2:
    st.info("Masukkan/upload minimal 2 kalimat untuk melakukan perhitungan.")
    st.stop()

matrix, token_sets = build_binary_matrix(valid_sentences)
pair_results = calculate_all_pairs(matrix)

# =========================================================
# Ringkasan
# =========================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Kalimat", len(valid_sentences))
with col2:
    st.metric("Jumlah Kata Unik", len(matrix))
with col3:
    st.metric("Jumlah Pasangan", len(pair_results))

# =========================================================
# Tab
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Data Kalimat", "🔢 Matriks Biner", "📊 Hasil Cosine", "🧮 Detail Perhitungan"]
)

with tab1:
    st.subheader("Data Kalimat")

    sentence_df = pd.DataFrame(
        [
            {
                "Kalimat": sid,
                "Teks": text,
                "Kata Unik": len(token_sets[sid]),
                "Norm": f"√{len(token_sets[sid])} = {math.sqrt(len(token_sets[sid])):.{decimals}f}",
            }
            for sid, text in valid_sentences.items()
        ]
    )

    st.dataframe(sentence_df, use_container_width=True, hide_index=True)

    st.markdown("### Token setiap kalimat")
    token_cols = st.columns(min(3, len(valid_sentences)))
    for i, (sid, tokens) in enumerate(token_sets.items()):
        with token_cols[i % len(token_cols)]:
            st.markdown(f"**{sid}**")
            st.write(", ".join(sorted(tokens)))

with tab2:
    st.subheader("Matriks Biner Kata–Kalimat")
    st.caption("1 berarti kata muncul pada kalimat; 0 berarti kata tidak muncul.")

    display_matrix = matrix.copy()
    display_matrix.index.name = "Kata"
    st.dataframe(display_matrix, use_container_width=True)

    totals = matrix.sum(axis=0).astype(int)
    st.markdown("### Total kata per kalimat")

    total_cols = st.columns(len(totals))
    for col, (sid, total) in zip(total_cols, totals.items()):
        with col:
            st.metric(sid, int(total), help=f"|{sid}| = √{int(total)}")

with tab3:
    st.subheader("Hasil Cosine Similarity")

    result_display = pair_results.copy()
    result_display["Kata Sama"] = result_display["common_words"].apply(
        lambda x: ", ".join(x)
    )
    result_display["Cosine Similarity"] = result_display["similarity"].round(decimals)
    result_display["Kemiripan (%)"] = (
        result_display["similarity"] * 100
    ).round(2)

    result_display = result_display[
        ["s1", "s2", "Kata Sama", "dot_product", "Cosine Similarity", "Kemiripan (%)"]
    ].rename(
        columns={
            "s1": "Kalimat X",
            "s2": "Kalimat Y",
            "dot_product": "X · Y",
        }
    )

    st.dataframe(result_display, use_container_width=True, hide_index=True)

    st.markdown("### Pasangan paling mirip")
    best = pair_results.loc[pair_results["similarity"].idxmax()]
    st.success(
        f"**{best['s1']} dan {best['s2']}** memiliki nilai cosine similarity "
        f"tertinggi: **{best['similarity']:.{decimals}f}** "
        f"({best['similarity'] * 100:.2f}%)."
    )

    st.markdown("### Urutan kemiripan")
    ranking = pair_results.sort_values("similarity", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking_display = ranking[["s1", "s2", "similarity"]].copy()
    ranking_display["similarity"] = ranking_display["similarity"].round(decimals)
    ranking_display.index.name = "Peringkat"
    st.dataframe(ranking_display, use_container_width=True)

with tab4:
    st.subheader("Detail Perhitungan")

    for _, row in pair_results.iterrows():
        s1, s2 = row["s1"], row["s2"]
        count_x = int(matrix[s1].sum())
        count_y = int(matrix[s2].sum())
        common_count = int(row["dot_product"])

        st.markdown(f"#### {s1} × {s2}")
        st.write(
            f"**Kata yang sama ({common_count}):** "
            + (", ".join(row["common_words"]) if row["common_words"] else "Tidak ada")
        )

        st.latex(
            rf"""
            \mathrm{{Cos}}({s1},{s2})
            =
            \frac{{{common_count}}}
            {{\sqrt{{{count_x}}}\times\sqrt{{{count_y}}}}}
            =
            \frac{{{common_count}}}
            {{{math.sqrt(count_x * count_y):.{decimals}f}}}
            =
            {row["similarity"]:.{decimals}f}
            """
        )

        st.caption(
            f"|{s1}| = √{count_x} = {math.sqrt(count_x):.{decimals}f}  •  "
            f"|{s2}| = √{count_y} = {math.sqrt(count_y):.{decimals}f}"
        )

        st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding-top:1rem;">
        Sistem Cosine Similarity • Python + Streamlit • Data Mining
    </div>
    """,
    unsafe_allow_html=True,
)
