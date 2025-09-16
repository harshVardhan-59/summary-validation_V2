import yaml
import json
import re
import requests
from pathlib import Path
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from summac.model_summac import SummaCZS

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    return text

def extract_identifiers(text):
        """
        Extract IDs, numbers, and tokens like 191754.002, years, deal IDs, etc.
        """
        return re.findall(r"[A-Za-z0-9\.\-_]+", text)

class GroundTruthGenerator:
    def __init__(self, sql_result):
        self.sql_result = sql_result

    def generate(self):
        count = len(self.sql_result)
        return f"The query returned {count} rights records."



class ForwardValidator:
    def __init__(self):
        print("[ForwardValidator] Loading factual consistency model (SummaCZS with RoBERTa-MNLI backbone)...")
        self.summac = SummaCZS(granularity="sentence", model_name="mnli", device="mps")

        print("[ForwardValidator] Loading embedding model (bge-large-en-v1.5)...")
        self.bge = SentenceTransformer("BAAI/bge-large-en-v1.5")



    def factual_consistency(self, summary, ref_summary):
        summary_norm = normalize(summary)
        ref_summary_norm = normalize(ref_summary)
        scores = self.summac.score([ref_summary_norm], [summary_norm], return_sent_scores=True)
        if not scores or "scores" not in scores:
            return 0.0
        raw_score = float(scores["scores"][0])       # in [-1, 1]
        normalized = (raw_score + 1) / 2             # maps to [0, 1]
        # Debug print
        print("[DEBUG] SummaC raw:", raw_score, "normalized:", normalized)
        return normalized

    def intent_alignment(self, summary, user_intent):
        # --- 1. Semantic similarity (embeddings) ---
        emb1 = self.bge.encode(summary, convert_to_tensor=True)
        emb2 = self.bge.encode(user_intent, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(emb1, emb2).cpu().item())

        # --- 2. Identifier overlap (property IDs, years, deal IDs) ---
        ids_intent = set(extract_identifiers(user_intent))
        ids_summary = set(extract_identifiers(summary))

        overlap = ids_intent.intersection(ids_summary)
        has_overlap = len(overlap) > 0

        # --- 3. Hybrid score ---
        score = semantic_score
        if has_overlap:
            # Boost: ensure at least moderate alignment
            score = max(score, 0.85)

        return score

    def answerability(self, summary, questions):
        qa_results = {}
        scores = []
        for question in questions:
            # Simple heuristic: check if any keyword from question is in summary
            keywords = [w for w in re.findall(r"\w+", question) if len(w) > 2]
            found = any(kw.lower() in summary.lower() for kw in keywords)
            qa_results[question] = {"answer": "Found" if found else "Not found", "score": 1.0 if found else 0.0}
            scores.append(1.0 if found else 0.0)
        qa_results["average_score"] = sum(scores) / len(scores) if scores else 0.0
        return qa_results




# class BackwardValidator:
#     def __init__(self):
#         print("[BackwardValidator] Loading entailment model (DeBERTa MNLI)...")
#         self.entailment_model = pipeline("text-classification", model="microsoft/deberta-large-mnli")

#         print("[BackwardValidator] Loading TAPEX model (for SQL-to-text)...")
#         self.tapex = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large")
#         self.tapex_tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large")

#     def clause_entailment(self, summary, sql_text):
#         return self.entailment_model(f"{sql_text} </s> {summary}")

#     def contradiction_detection(self, summary):
#         return self.entailment_model(summary)

#     def sql_to_text_similarity(self, summary, sql_text):
#         emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
#         emb1 = emb_model.encode(summary, convert_to_tensor=True)
#         emb2 = emb_model.encode(sql_text, convert_to_tensor=True)
#         return util.cos_sim(emb1, emb2).item()

class BackwardValidator:
    def __init__(self):
        print("[BackwardValidator] Loading entailment model (DeBERTa MNLI)...")
        self.entailment_model = pipeline(
            "text-classification",
            model="microsoft/deberta-large-mnli"
        )

        print("[BackwardValidator] Loading TAPEX model (for SQL-to-text)...")
        self.tapex = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large")
        self.tapex_tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large")

        print("[BackwardValidator] Loading embedding model (bge-large-en-v1.5)...")
        self.emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def clause_entailment(self, summary, sql_text):
        """
        Check if the SQL text logically entails the summary.
        Uses DeBERTa MNLI with proper premise-hypothesis format.
        """
        result = self.entailment_model(
            {"text": sql_text, "text_pair": summary}
        )
        return result  # returns label + score

    def contradiction_detection(self, summary, sql_text):
        """
        Check if the SQL contradicts the summary.
        Again requires premise-hypothesis format.
        """
        result = self.entailment_model(
            {"text": sql_text, "text_pair": summary}
        )
        # Just return full result (ENTAILMENT, NEUTRAL, CONTRADICTION with scores)
        return result

    def sql_to_text_similarity(self, summary, sql_text):
        """
        Compute semantic similarity between SQL and summary.
        Note: SQL is structured → similarity will be lower.
        """
        emb1 = self.emb_model.encode(summary, convert_to_tensor=True)
        emb2 = self.emb_model.encode(sql_text, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).cpu().item())


class InputProvider:
    def __init__(self, config_path="config.yaml", api_url=None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.api_url = api_url
        self.last_sources = {}  # Track source for each key

    def get_value(self, key):
        # Try API first
        if self.api_url:
            try:
                response = requests.get(f"{self.api_url}/{key}", timeout=5)
                if response.status_code == 200:
                    value = response.json().get(key)
                    if value:
                        print(f"[InputProvider] Got '{key}' from API.")
                        self.last_sources[key] = "API"
                        return value
            except Exception as e:
                print(f"[InputProvider] API error for '{key}': {e}")
        # Fallback to config
        print(f"[InputProvider] Using '{key}' from config.")
        self.last_sources[key] = "Config"
        return self.config.get(key)

    def get_all(self, keys):
        return {key: self.get_value(key) for key in keys}

    def get_sources(self):
        # Returns a dict of key: source ("API" or "Config")
        return self.last_sources

class SummaryValidator:
    def __init__(self, config_path="config.yaml", api_url=None):
        self.input_provider = InputProvider(config_path=config_path, api_url=api_url)
        self.num_questions = self.input_provider.get_value("num_questions") or 5
        self.forward = ForwardValidator()
        self.backward = BackwardValidator()
        self.thresholds = self.input_provider.get_value("thresholds") or {"good": 0.8, "moderate": 0.6}


    def status(self, score):
        if score >= self.thresholds["good"]:
            return "✅ Good"
        elif score >= self.thresholds["moderate"]:
            return "⚠️ Moderate"
        else:
            return "❌ Poor"


    def interpret_results(self, raw_results):
        answerability = raw_results["forward_validation"]["answerability"]
        # Remove aggregate score if present
        questions = [q for q in answerability.keys() if q != "average_score"]

        return {
            "Data Source": raw_results.get("data_source", "Config"),
            "Ground Truth Summary": raw_results["truth_summary"],

            "Forward Validation": {
                "Factual Consistency": {
                    "Result": self.status(raw_results["forward_validation"]["factual_consistency"]),
                    "Confidence": round(raw_results["forward_validation"]["factual_consistency"], 2)
                },
                "Intent Alignment": {
                    "Result": self.status(raw_results["forward_validation"]["intent_alignment"]),
                    "Similarity Score": round(raw_results["forward_validation"]["intent_alignment"], 2)
                },
                "Answerability": {
                    "Questions": [
                        {
                            "Question": q,
                            "Answer": answerability[q]["answer"],
                            "Score": round(answerability[q]["score"], 2)
                        }
                        for q in questions
                    ],
                    "Average Score": round(answerability.get("average_score", 0.0), 2)
                }
            },

            # "Backward Validation": {
            #     "Clause Entailment": {
            #         "Result": raw_results["backward_validation"]["clause_entailment"][0]["label"],
            #         "Confidence": round(raw_results["backward_validation"]["clause_entailment"][0]["score"], 2)
            #     },
            #     "Contradiction Check": {
            #         "Result": raw_results["backward_validation"]["contradiction_detection"][0]["label"],
            #         "Confidence": round(raw_results["backward_validation"]["contradiction_detection"][0]["score"], 2)
            #     },
            #     "SQL-to-Text Match": {
            #         "Similarity": round(raw_results["backward_validation"]["sql_to_text_similarity"], 2),
            #         "Interpretation": self.status(raw_results["backward_validation"]["sql_to_text_similarity"])
            #     }
            # }
            "Backward Validation": {
                "Clause Entailment": {
                    "Result": raw_results["backward_validation"]["clause_entailment"]["label"],
                    "Confidence": round(raw_results["backward_validation"]["clause_entailment"]["score"], 2)
                },
                "Contradiction Check": {
                    "Result": raw_results["backward_validation"]["contradiction_detection"]["label"],
                    "Confidence": round(raw_results["backward_validation"]["contradiction_detection"]["score"], 2)
                },
                "SQL-to-Text Match": {
                    "Similarity": round(raw_results["backward_validation"]["sql_to_text_similarity"], 2),
                    "Interpretation": self.status(raw_results["backward_validation"]["sql_to_text_similarity"])
                }
            }

        }

    def run(self):
        keys = ["user_intent", "sql_result", "sql_text", "generated_summary", "ground_truth_summary", "questions"]
        inputs = self.input_provider.get_all(keys)
        sources = self.input_provider.get_sources()
        # Determine overall source (API if any key is from API, else Config)
        overall_source = "API" if "API" in sources.values() else "Config"

        user_intent = inputs["user_intent"]
        sql_result = inputs["sql_result"]
        sql_text = inputs["sql_text"]
        generated_summary = inputs["generated_summary"]
        truth_summary = inputs.get("ground_truth_summary")
        questions = inputs.get("questions", [])
        print("Loaded questions:", questions)
        
        forward_results = {
            "factual_consistency": self.forward.factual_consistency(generated_summary, truth_summary),
            "intent_alignment": self.forward.intent_alignment(generated_summary, user_intent),
            "answerability": self.forward.answerability(generated_summary, questions),
        } 
        backward_results = {
            "clause_entailment": self.backward.clause_entailment(generated_summary, sql_text),
            "contradiction_detection": self.backward.contradiction_detection(generated_summary, sql_text),
            "sql_to_text_similarity": self.backward.sql_to_text_similarity(generated_summary, sql_text),
        }

        raw_results = {
            "data_source": overall_source,  # <-- Add this line
            "truth_summary": truth_summary,
            "forward_validation": forward_results,
            "backward_validation": backward_results,
        }

        return self.interpret_results(raw_results)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    api_url = config.get("api_url", None)

    validator = SummaryValidator(config_path="config.yaml", api_url=api_url)
    result = validator.run()
    # ✅ Pretty-print JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # ✅ Save JSON to file
    output_path = Path("validation_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nValidation results saved to: {output_path.resolve()}")

