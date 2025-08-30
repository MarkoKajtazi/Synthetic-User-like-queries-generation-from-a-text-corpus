from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import fitz
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed



def summarize(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=128,
        min_length=50,
        no_repeat_ngram_size=2,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def extract_paragraphs(pdf_path, min_length=300, size_tolerance=0.5):
    doc = fitz.open(pdf_path)

    all_sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    all_sizes.append(round(span["size"], 1))
    if not all_sizes:
        return []

    body_size = Counter(all_sizes).most_common(1)[0][0]

    paragraphs = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            spans = []
            text = ""
            for line in block.get("lines", []):
                for span in line["spans"]:
                    spans.append(round(span["size"], 1))
                    text += span["text"]
                text += "\n"
            if not spans:
                continue

            avg_size = sum(spans) / len(spans)
            if abs(avg_size - body_size) > size_tolerance:
                continue

            for para in text.split("\n\n"):
                p = para.strip()
                if len(p) >= min_length and not p.isupper():
                    paragraphs.append(p)

    return paragraphs

def get_paragraphs():
    paragraphs = extract_paragraphs("../Data/First Project Assignment - Research based design report.pdf")

    device = torch.device("cpu")
    model_name = "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    summaries = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_para = {
            executor.submit(summarize, para, tokenizer, model): para
            for para in paragraphs
        }
        for future in as_completed(future_to_para):
            try:
                summ = future.result()
                summaries.append(summ)
            except Exception as exc:
                para = future_to_para[future]
                print(f"Error summarizing paragraph (len={len(para)}): {exc}")

    all_text = " ".join(paragraphs)
    overall_summ = summarize(all_text, tokenizer, model)
    summaries.append(overall_summ)

    return summaries

# get_paragraphs()
paragraphs = extract_paragraphs("../Data/First Project Assignment - Research based design report.pdf")
for para in paragraphs:
    print(f"{para}\n")