import yaml
import json
from pathlib import Path
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util


class GroundTruthGenerator:
    def __init__(self, sql_result):
        self.sql_result = sql_result

    def generate(self):
        count = len(self.sql_result)
        return f"The query returned {count} rights records."


class ForwardValidator:
    def __init__(self):
        print("[ForwardValidator] Loading factual consistency model (DeBERTa MNLI)...")
        self.summaCZS = pipeline("text-classification", model="microsoft/deberta-large-mnli")

        print("[ForwardValidator] Loading embedding model (bge-large-en-v1.5)...")
        self.bge = SentenceTransformer("BAAI/bge-large-en-v1.5")

        print("[ForwardValidator] Loading QA model (roberta-base-squad2)...")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.qa_pipeline = pipeline("question-answering", model=self.qa_model, tokenizer=self.qa_tokenizer)

    def factual_consistency(self, summary, ref_summary):
        result = self.summaCZS(f"{ref_summary} </s> {summary}")
        return result[0]["score"]

    def intent_alignment(self, summary, user_intent):
        emb1 = self.bge.encode(summary, convert_to_tensor=True)
        emb2 = self.bge.encode(user_intent, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def answerability(self, summary):
        question = "What is returned?"
        result = self.qa_pipeline(question=question, context=summary)
        return {question: {"answer": result["answer"], "score": result["score"]}}


class BackwardValidator:
    def __init__(self):
        print("[BackwardValidator] Loading entailment model (DeBERTa MNLI)...")
        self.entailment_model = pipeline("text-classification", model="microsoft/deberta-large-mnli")

        print("[BackwardValidator] Loading TAPEX model (for SQL-to-text)...")
        self.tapex = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large")
        self.tapex_tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large")

    def clause_entailment(self, summary, sql_text):
        return self.entailment_model(f"{sql_text} </s> {summary}")

    def contradiction_detection(self, summary):
        return self.entailment_model(summary)

    def sql_to_text_similarity(self, summary, sql_text):
        emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        emb1 = emb_model.encode(summary, convert_to_tensor=True)
        emb2 = emb_model.encode(sql_text, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()


class SummaryValidator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.forward = ForwardValidator()
        self.backward = BackwardValidator()

        self.thresholds = self.config.get("thresholds", {"good": 0.8, "moderate": 0.6})

    def status(self, score):
        if score >= self.thresholds["good"]:
            return "✅ Good"
        elif score >= self.thresholds["moderate"]:
            return "⚠️ Moderate"
        else:
            return "❌ Poor"

    def interpret_results(self, raw_results):
        return {
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
                    "Question": "What is returned?",
                    "Answer": raw_results["forward_validation"]["answerability"]["What is returned?"]["answer"],
                    "Confidence": round(raw_results["forward_validation"]["answerability"]["What is returned?"]["score"], 2)
                }
            },

            "Backward Validation": {
                "Clause Entailment": {
                    "Result": raw_results["backward_validation"]["clause_entailment"][0]["label"],
                    "Confidence": round(raw_results["backward_validation"]["clause_entailment"][0]["score"], 2)
                },
                "Contradiction Check": {
                    "Result": raw_results["backward_validation"]["contradiction_detection"][0]["label"],
                    "Confidence": round(raw_results["backward_validation"]["contradiction_detection"][0]["score"], 2)
                },
                "SQL-to-Text Match": {
                    "Similarity": round(raw_results["backward_validation"]["sql_to_text_similarity"], 2),
                    "Interpretation": self.status(raw_results["backward_validation"]["sql_to_text_similarity"])
                }
            }
        }

    def run(self):
        user_intent = self.config["user_intent"]
        sql_result = self.config["sql_result"]
        sql_text = self.config["sql_text"]
        generated_summary = self.config["generated_summary"]

        truth_summary = GroundTruthGenerator(sql_result).generate()

        forward_results = {
            "factual_consistency": self.forward.factual_consistency(generated_summary, truth_summary),
            "intent_alignment": self.forward.intent_alignment(generated_summary, user_intent),
            "answerability": self.forward.answerability(generated_summary),
        }

        backward_results = {
            "clause_entailment": self.backward.clause_entailment(generated_summary, sql_text),
            "contradiction_detection": self.backward.contradiction_detection(generated_summary),
            "sql_to_text_similarity": self.backward.sql_to_text_similarity(generated_summary, sql_text),
        }

        raw_results = {
            "truth_summary": truth_summary,
            "forward_validation": forward_results,
            "backward_validation": backward_results,
        }

        return self.interpret_results(raw_results)


if __name__ == "__main__":
    validator = SummaryValidator()
    result = validator.run()

    # ✅ Pretty-print JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # ✅ Save JSON to file
    output_path = Path("validation_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nValidation results saved to: {output_path.resolve()}")

