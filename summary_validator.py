import logging
import json
import re
import requests
import yaml
import os
from pathlib import Path
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from summac.model_summac import SummaCZS
from nltk import sent_tokenize
from datetime import date
from DatabaseConn import DatabaseConn

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def normalize(text):
    """Lowercase and collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def clean_text(text):
    """Clean text for SummaC by removing special characters and extra formatting."""
    text = re.sub(r"[\n\t•-]", " ", text)  # Remove newlines, tabs, bullets
    text = re.sub(r"\s+", " ", text)  # Collapse spaces
    text = re.sub(r"[()[\]{}]", "", text)  # Remove parentheses, brackets
    text = text.strip()
    return text

def extract_identifiers(text):
    """Extract identifiers, preserving phrases and handling special characters."""
    # Match numbers, dates (DD-MM-YYYY or YYYY-MM-DD), and phrases with spaces or special chars
    pattern = r'\d{1,10}(?:\.\d+)?|\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2}|[\w\s&/-]+'
    identifiers = []
    for match in re.findall(pattern, text):
        match = match.strip()
        # Skip short or irrelevant matches
        if len(match) > 2 and not re.match(r"^\W+$", match):
            # Normalize dates to DD-MM-YYYY
            if re.match(r"\d{4}-\d{2}-\d{2}", match):
                match = f"{match[8:10]}-{match[5:7]}-{match[0:4]}"
            identifiers.append(normalize(match))
    return identifiers

def json_to_text(data, prefix=""):
    """Recursively convert JSON to natural language text for any schema."""
    sentences = []
    if isinstance(data, dict):
        if prefix.endswith("media_types"):
            for key in data.keys():
                sentences.append(f"{key.replace('_', ' ').title()} is a primary right.")
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}"
                sentences.extend(json_to_text(value, new_prefix))
        else:
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                sentences.extend(json_to_text(value, new_prefix))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_prefix = f"{prefix}[{i}]"
            sentences.extend(json_to_text(item, new_prefix))
    else:
        value_str = "None" if data is None else str(data)
        key_name = prefix.replace("_", " ").replace(".", " ").title()
        sentences.append(f"{key_name} is {value_str}.")
    return sentences

class GroundTruthGenerator:
    def __init__(self, sql_result):
        self.sql_result = sql_result

    def generate(self):
        count = len(self.sql_result) if isinstance(self.sql_result, list) else 1
        return f"The query returned {count} rights records."

class ForwardValidator:
    def __init__(self):
        logger.info("[ForwardValidator] Loading factual consistency model (SummaCZS with DeBERTa-MNLI backbone)...")
        device = "cpu"
        try:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            logger.info(f"[ForwardValidator] Using device: {device}")
        except Exception as e:
            logger.warning(f"[ForwardValidator] Failed to detect device: {e}. Defaulting to CPU.")
        
        # Initialize SummaCZS with a supported model (deberta-large-mnli)
        try:
            self.summac = SummaCZS(
                granularity="sentence",
                model_name="deberta-large-mnli",  # Changed from roberta-large-mnli
                device=device,
                tokenizer_kwargs={"truncation": True}  # Avoid deprecated truncation_strategy
            )
        except Exception as e:
            logger.error(f"[ForwardValidator] Failed to load SummaCZS: {e}. Falling back to CPU.")
            self.summac = SummaCZS(
                granularity="sentence",
                model_name="deberta-large-mnli",
                device="cpu",
                tokenizer_kwargs={"truncation": True}
            )
        
        logger.info("[ForwardValidator] Loading embedding model (bge-large-en-v1.5)...")
        self.bge = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        logger.info("[ForwardValidator] Loading QA model (distilbert-base-cased-distilled-squad)...")
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=device,
                token=os.environ.get("HUGGINGFACE_TOKEN")
            )
        except Exception as e:
            logger.error(f"[ForwardValidator] Failed to load QA model: {e}")
            raise

    def sql_to_reference_text(self, sql_result):
        """Convert SQL result (JSON) to text dynamically, handling any schema."""
        if isinstance(sql_result, str):
            return sql_result
        if not isinstance(sql_result, list):
            sql_result = [sql_result]
        lines = []
        for row in sql_result:
            lines.extend(json_to_text(row))
        reference_text = "\n".join(lines)
        logger.info(f"[ForwardValidator] Generated reference text:\n{reference_text}")
        return reference_text

    def sql_to_summary_text(self, sql_result):
        """Generate a summarized reference text for factual consistency checking."""
        if not isinstance(sql_result, list):
            sql_result = [sql_result]
        sentences = []
        for row in sql_result:
            sentences.append(f"Total matching title count: {row.get('total_title_count', 1)}")
            sentences.append(f"Title selected: {row.get('program_name')} ({row.get('program_type')}), ID: {row.get('title_source_id')} ({row.get('title_source')})")
            sentences.append(f"Incorporated Deal: {row.get('deal_name')} ({row.get('deal_type')}), ID: {row.get('deal_id')} ({row.get('rights_source')}), {row.get('hive_deal_id')} (HIVE)")
            sentences.append(f"Primary Parties: {', '.join(row.get('deal_primary_parties', []))}")
            # Convert dates to DD-MM-YYYY for consistency
            term_from = row.get('term_from')
            term_to = row.get('term_to')
            if isinstance(term_from, date):
                term_from = term_from.strftime("%d-%m-%Y")
            elif isinstance(term_from, str) and re.match(r"\d{4}-\d{2}-\d{2}", term_from):
                term_from = f"{term_from[8:10]}-{term_from[5:7]}-{term_from[0:4]}"
            if isinstance(term_to, date):
                term_to = term_to.strftime("%d-%m-%Y")
            elif isinstance(term_to, str) and re.match(r"\d{4}-\d{2}-\d{2}", term_to):
                term_to = f"{term_to[8:10]}-{term_to[5:7]}-{term_to[0:4]}"
            sentences.append(f"License term: {term_from} to {term_to}")
            media_types = row.get('media_types', {})
            primary_rights = [k for k in media_types.keys()]
            sentences.append(f"Primary rights: {', '.join(primary_rights)}")
            ancillary = []
            for mt, attrs in media_types.items():
                if attrs.get("Simulcast"): ancillary.append(f"Simulcast ({mt})")
                if attrs.get("Start Over"): ancillary.append(f"Start Over ({mt})")
                if attrs.get("Catch-Up"): ancillary.append(f"Catch-Up ({mt})")
                if attrs.get("Temporary Download"): ancillary.append(f"Temporary Download ({mt})")
            if ancillary:
                sentences.append(f"Ancillary rights: {', '.join(ancillary)}")
            sentences.append(f"Territories: {row.get('countries')}")
            sentences.append(f"Brands: {', '.join(row.get('brands', [])[:10])} ...")
            sentences.append(f"Languages: {', '.join(media_types.get('Pay TV', {}).get('Allowed Languages', ['None']))}")
            sentences.append(f"Exclusivity: {row.get('exclusivity', 'None')}")
        reference_text = "\n".join(sentences)
        logger.info(f"[ForwardValidator] Generated summary reference text:\n{reference_text}")
        return reference_text

    def rule_based_check(self, summary, sql_response):
        """
        Dynamic rule-based factual consistency check as a fallback, avoiding hardcoding.
        """
        logger.info("[ForwardValidator] Performing dynamic rule-based factual consistency check...")

        if not summary or not sql_response:
            logger.warning("[ForwardValidator] Empty summary or SQL response; returning poor result")
            return {"Result": "❌ Poor", "Confidence": 0.0, "Claims": []}

        claims = self.extract_claims(summary)
        if not claims:
            logger.warning("[ForwardValidator] No claims extracted from summary")
            return {"Result": "❌ Poor", "Confidence": 0.0, "Claims": []}

        if not isinstance(sql_response, list) or not sql_response:
            logger.warning("[ForwardValidator] Invalid SQL response format")
            return {"Result": "❌ Poor", "Confidence": 0.0, "Claims": []}
        sql_data = sql_response[0]

        claim_results = []
        scores = []

        for idx, claim in enumerate(claims, 1):
            claim_lower = claim["text"].lower()
            score = 0.0
            verdict = "inaccurate"
            matched = False

            try:
                # Title count claim
                if "total matching title count" in claim_lower:
                    expected = str(sql_data.get("total_title_count", 0))
                    if expected in claim["text"]:
                        score = 1.0
                        matched = True

                # Title selected claim
                elif "title selected" in claim_lower:
                    program_name = sql_data.get("program_name", "").lower()
                    title_source_id = str(sql_data.get("title_source_id", "")).replace(".0", "")
                    title_source = sql_data.get("title_source", "").lower()
                    if program_name in claim_lower and title_source_id in claim["text"] and title_source in claim_lower:
                        score = 1.0
                        matched = True

                # Deal claim (general for any deal name)
                elif "commission" in claim_lower or "deal" in claim_lower:
                    deal_name = sql_data.get("deal_name", "").lower()
                    deal_id = str(sql_data.get("deal_id", "")).replace(".0", "")
                    hive_deal_id = str(sql_data.get("hive_deal_id", "")).replace(".0", "")
                    primary_parties = [p.lower() for p in sql_data.get("deal_primary_parties", [])]
                    if deal_name in claim_lower and deal_id in claim["text"] and hive_deal_id in claim["text"] and all(p in claim_lower for p in primary_parties[:2]):
                        score = 1.0
                        matched = True

                # Primary rights claim (e.g., "Pay TV: Yes")
                elif ": yes" in claim_lower and any(right in claim_lower for right in ["pay tv", "free tv", "ppv", "fast", "tvod", "stb vod", "svod", "download to own"]):
                    right_name = re.search(r"([a-z\s]+):\s*yes", claim_lower, re.IGNORECASE)
                    if right_name:
                        right_key = right_name.group(1).strip().title()
                        if right_key.lower() in [k.lower() for k in sql_data.get("media_types", {}).keys()]:
                            score = 1.0
                            matched = True

                # Ancillary rights claim (e.g., "Simulcast: FAST (Yes)")
                elif ": yes" in claim_lower and " (" in claim_lower:
                    ancillary_match = re.match(r"(\w+\s*(?:\w+\s*)*):\s*(\w+\s*(?:\w+\s*)*)\s*\(Yes\)", claim["text"], re.IGNORECASE)
                    if ancillary_match:
                        right = ancillary_match.group(1).title()
                        mt = ancillary_match.group(2).title()
                        attrs = sql_data.get("media_types", {}).get(mt, {})
                        if right.lower() in attrs and attrs[right.lower()] == True:
                            score = 1.0
                            matched = True

                # Territories claim
                elif "territories" in claim_lower:
                    if sql_data.get("countries", "").lower() == "world wide" and "world wide" in claim_lower:
                        score = 1.0
                        matched = True

                # Brands claim
                elif "brands" in claim_lower:
                    brands = sql_data.get("brands", [])
                    if any(brand.lower() in claim_lower for brand in brands[:10]):  # Check at least some brands
                        score = 1.0
                        matched = True

                # Languages claim
                elif "languages" in claim_lower:
                    if "all languages" in claim_lower and all("All Languages" in sql_data.get("media_types", {}).get(mt, {}).get("Allowed Languages", []) for mt in sql_data.get("media_types", {}).keys()):
                        score = 1.0
                        matched = True

                # Earliest term_from date claim
                elif "earliest term_from date" in claim_lower:
                    term_from = sql_data.get("term_from", "").replace("-", "")
                    if term_from in claim["text"].replace("-", ""):
                        score = 1.0
                        matched = True

                # Latest term_to date claim
                elif "latest term_to date" in claim_lower:
                    term_to = sql_data.get("term_to", "").replace("-", "")
                    if term_to in claim["text"].replace("-", ""):
                        score = 1.0
                        matched = True

                # No overlaps or gaps in term windows claim
                elif "no overlaps or gaps" in claim_lower:
                    if not sql_data.get("estimated_term_from", True) and not sql_data.get("estimated_term_to", True):
                        score = 1.0
                        matched = True

            except Exception as e:
                logger.warning(f"[ForwardValidator] Error validating claim '{claim}': {e}")

            if not matched:
                logger.warning(f"[ForwardValidator] No validation rule matched for claim: {claim}")

            claim_results.append({
                "claim_id": idx,
                "sentence": claim["text"],
                "score": score,
                "verdict": "accurate" if score > 0.5 else "inaccurate"
            })
            scores.append(score)

        confidence = sum(scores) / len(scores) if scores else 0.0
        result = "✅ Good" if confidence >= 0.8 else "⚠️ Moderate" if confidence >= 0.5 else "❌ Poor"

        logger.info(f"[ForwardValidator] Rule-based factual consistency: confidence={confidence}, result={result}")

        return {
            "Result": result,
            "Confidence": round(confidence, 2),
            "Claims": claim_results
        }

    def extract_claims(self, summary):
        sentences = sent_tokenize(summary)
        claims = []
        for s in sentences:
            if any(keyword in s.lower() for keyword in [
                "total matching title count", "title selected", "deal", "commission",
                ": yes", "territories", "brands", "languages", "term_from", "term_to", "no overlaps"
            ]):
                claims.append({"text": s.strip()})
        return claims

    def factual_consistency(self, summary, reference):
        """
        Check factual consistency using SummaCZS, with dynamic rule-based check as fallback.
        """
        logger.info("[ForwardValidator] Checking factual consistency with SummaCZS...")
        if not summary or not reference:
            logger.warning("[ForwardValidator] Empty summary or reference; returning poor result")
            return {"Result": "❌ Poor", "Confidence": 0.0, "Claims": []}

        try:
            summary_sentences = sent_tokenize(summary)
            reference_sentences = sent_tokenize(reference)
            if not summary_sentences or not reference_sentences:
                logger.warning("[ForwardValidator] No sentences extracted; returning poor result")
                return {"Result": "❌ Poor", "Confidence": 0.0, "Claims": []}

            # Run SummaCZS
            summac_score = self.summac.score(
                originals=reference_sentences,
                generated=summary_sentences
            )
            logger.info(f"[ForwardValidator] SummaCZS score: {summac_score}")

            # Extract claims from summary
            claims = self.extract_claims(summary)
            if not claims:
                logger.warning("[ForwardValidator] No claims extracted from summary")
                return {"Result": "❌ Poor", "Confidence": 0.0, "Claims": []}

            # Map scores to claims
            claim_results = []
            scores = summac_score["scores"]
            for idx, (claim, score) in enumerate(zip(claims, scores[:len(claims)]), 1):
                verdict = "accurate" if score >= 0.5 else "inaccurate"
                claim_results.append({
                    "claim_id": idx,
                    "sentence": claim["text"],
                    "score": round(score, 2),
                    "verdict": verdict
                })

            # Calculate overall result
            confidence = sum(scores[:len(claims)]) / len(claims) if claims else 0.0
            result = "✅ Good" if confidence >= 0.8 else "⚠️ Moderate" if confidence >= 0.5 else "❌ Poor"

            return {
                "Result": result,
                "Confidence": round(confidence, 2),
                "Claims": claim_results
            }
        except Exception as e:
            logger.error(f"[ForwardValidator] SummaCZS failed: {e}. Falling back to rule-based check.")
            return self.rule_based_check(summary, reference)

    def intent_alignment(self, summary, user_intent):
        if not summary or not user_intent or not isinstance(summary, str) or not isinstance(user_intent, str):
            logger.warning("[ForwardValidator] Empty or invalid summary/user_intent; returning score 0.0")
            return 0.0
        emb1 = self.bge.encode(summary, convert_to_tensor=True)
        emb2 = self.bge.encode(user_intent, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(emb1, emb2).cpu().item())
        ids_intent = set(extract_identifiers(normalize(clean_text(user_intent))))
        ids_summary = set(extract_identifiers(normalize(clean_text(summary))))
        overlap = ids_intent.intersection(ids_summary)
        has_overlap = len(overlap) > 0
        score = semantic_score
        if has_overlap:
            score = max(score, 0.85)
        return score

    def answerability(self, summary, questions):
        """
        Perform QA on the summary to evaluate answerability. (Unchanged)
        """
        if not summary or not isinstance(summary, str):
            logger.warning("[ForwardValidator] Empty or invalid summary; returning empty answerability results")
            return {"average_score": 0.0}

        logger.info(f"[ForwardValidator] Original generated_summary: {summary}")
        summary_clean = self.flatten_summary_for_qa(summary)

        qa_results = {}
        scores = []

        for question in questions:
            question_clean = re.sub(r"\bhas\b", "have", question)
            question_clean = question_clean.strip("?") + "?"
            try:
                result = self.qa_pipeline(
                    question=question_clean,
                    context=summary_clean,
                    max_answer_len=30,
                    top_k=1
                )
                answer_text = result["answer"].replace("\n", " ").strip()
                score = float(result["score"])
                logger.info(f"[ForwardValidator] QA result for '{question_clean}': answer='{answer_text}', score={score}")

                if "does" in question_clean.lower() and "include" in question_clean.lower():
                    media_type_match = re.search(r"include\s+([A-Za-z\s]+?)\s+as\s+a\s+primary\s+right", question_clean, re.IGNORECASE)
                    if media_type_match:
                        media_type = media_type_match.group(1).strip().lower()
                        if media_type in answer_text.lower() or media_type in summary_clean.lower():
                            answer_text = "Yes"
                            score = max(score, 0.8)
                        elif "none" in answer_text.lower() or not answer_text:
                            answer_text = "No"
                        else:
                            answer_text = "Information not available in summary"
                    else:
                        media_types = ["pay tv", "free tv", "ppv", "fast", "tvod", "stb vod", "svod", "download to own"]
                        if any(right.lower() in answer_text.lower() for right in media_types):
                            answer_text = "Yes"
                            score = max(score, 0.8)
                        elif "none" in answer_text.lower() or not answer_text:
                            answer_text = "No"
                        else:
                            answer_text = "Information not available in summary"

                if not answer_text or answer_text.lower() in ["none", "not found"] or "incorporated deal" in answer_text.lower():
                    answer_text = "Information not available in summary"
                qa_results[question] = {"answer": answer_text, "score": round(score, 2)}
                scores.append(score)
            except Exception as e:
                logger.error(f"[ForwardValidator] QA failed for '{question}': {e}")
                qa_results[question] = {"answer": "Error", "score": 0.0}
                scores.append(0.0)

        qa_results["average_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        return qa_results

    def flatten_summary_for_qa(self, summary):
        """Flatten the generated summary into short QA-friendly sentences."""
        if not summary or not isinstance(summary, str):
            logger.warning("[ForwardValidator] Empty or invalid summary; returning empty string")
            return ""

        # Replace placeholder dates
        summary = re.sub(r"\b12-31-9999\b", "Not specified", summary)
        # Remove bullets, indentation, tabs
        summary = re.sub(r"[-•\t]", "", summary)
        # Split lines
        lines = [line.strip() for line in summary.split("\n") if line.strip()]
        flat_sentences = []
        deal_count = 0
        in_deals_section = False
        in_primary_rights_section = False
        in_ancillary_rights_section = False
        primary_rights = []
        ancillary_rights = []

        for line in lines:
            # Detect section transitions
            if "List of incorporated Deals:" in line:
                in_deals_section = True
                in_primary_rights_section = False
                in_ancillary_rights_section = False
                continue
            elif "Primary rights:" in line:
                in_deals_section = False
                in_primary_rights_section = True
                in_ancillary_rights_section = False
                continue
            elif "Ancillary rights:" in line:
                in_deals_section = False
                in_primary_rights_section = False
                in_ancillary_rights_section = True
                continue
            elif any(line.startswith(s) for s in ["Territories:", "Brands:", "Languages:", "Notes:", "Exclusivity:", "Earliest term_from date:", "Latest term_to date:"]):
                in_deals_section = False
                in_primary_rights_section = False
                in_ancillary_rights_section = False

            # Count deals in deals section
            if in_deals_section and line.startswith("  "):
                deal_count += 1
                flat_sentences.append(f"Property has incorporated deal: {line.strip()}.")
            # Handle primary rights
            elif in_primary_rights_section and line.strip():
                right_match = re.match(r"(\w+\s*(?:\w+\s*)*):\s*Yes", line.strip(), re.IGNORECASE)
                if right_match:
                    primary_rights.append(right_match.group(1))
            # Handle ancillary rights
            elif in_ancillary_rights_section and line.strip():
                ancillary_match = re.match(r"(\w+\s*(?:\w+\s*)*):\s*(\w+\s*(?:\w+\s*)*)\s*\(Yes\)", line.strip(), re.IGNORECASE)
                if ancillary_match:
                    ancillary_rights.append(f"{ancillary_match.group(1)} ({ancillary_match.group(2)})")
            # Handle other fields
            elif "Total matching title count" in line:
                count_match = re.search(r"Total matching title count:\s*(\d+)", line)
                if count_match:
                    flat_sentences.append(f"Property has {count_match.group(1)} matching title(s).")
            elif line.startswith("Title selected:"):
                flat_sentences.append(line.replace("Title selected:", "Title selected is").strip() + ".")
            elif line.startswith("License term:"):
                flat_sentences.append(line.replace("License term:", "License term is").strip() + ".")
            elif "Territories:" in line:
                terr_match = re.findall(r"Territories: (.+)", line)
                if terr_match:
                    flat_sentences.append("Territories: " + terr_match[0].strip() + ".")
            elif "Brands:" in line:
                brands_match = re.findall(r"Brands: (.+)", line)
                if brands_match:
                    flat_sentences.append("Brands: " + brands_match[0].strip() + ".")
            elif "Languages:" in line:
                lang_match = re.findall(r"Languages: (.+)", line)
                if lang_match:
                    flat_sentences.append("Languages: " + lang_match[0].strip() + ".")
            elif line.startswith("Notes:") or line.startswith("Exclusivity:"):
                flat_sentences.append(line.strip() + ".")
            elif line.startswith("Earliest term_from date:"):
                flat_sentences.append(line.strip() + ".")
            elif line.startswith("Latest term_to date:"):
                flat_sentences.append(line.strip() + ".")

        # Add deal count if detected
        if deal_count > 0:
            flat_sentences.insert(0, f"Property has {deal_count} incorporated deal(s).")

        # Add primary rights
        if primary_rights:
            flat_sentences.append(f"Primary rights include: {', '.join(primary_rights)}.")
        else:
            flat_sentences.append("Primary rights: None specified.")

        # Add ancillary rights
        if ancillary_rights:
            flat_sentences.append(f"Ancillary rights include: {', '.join(ancillary_rights)}.")
        else:
            flat_sentences.append("Ancillary rights: None specified.")

        flat_summary = " ".join(flat_sentences)
        flat_summary = re.sub(r"\s+", " ", flat_summary)
        logger.info(f"[ForwardValidator] Flattened summary for QA: {flat_summary}")
        return flat_summary.strip()


class InputProvider:
    def __init__(self, config, api_url=None):
        self.config = config
        self.api_url = api_url
        self._api_cache = None
        self.last_sources = {}

    def fetch_api_response(self):
        if self._api_cache is not None:
            return self._api_cache
        if not self.api_url:
            logger.error("[InputProvider] No api_url provided; returning empty response")
            self._api_cache = {}
            return self._api_cache
        try:
            user_intent = self.config.get("user_intent", "")
            if not user_intent:
                logger.warning("[InputProvider] No user_intent in config; using empty prompt")
            response = requests.post(self.api_url, json={"prompt": user_intent})
            response.raise_for_status()
            self._api_cache = response.json()
            logger.info(f"[InputProvider] Full API response:\n{json.dumps(self._api_cache, indent=2, default=str)}")
        except Exception as e:
            logger.error(f"[InputProvider] Failed to fetch API response: {e}")
            self._api_cache = {}
        return self._api_cache

    def get_value(self, key, override_value=None, api_data=None):
        if override_value is not None:
            self.last_sources[key] = "UserInput"
            return override_value
        if api_data is None:
            api_data = self.fetch_api_response()
        key_mapping = {
            "sql_text": "sqlGenerated",
            "generated_summary": "summaryGenerated",
            "user_intent": "prompt"
        }
        if key in key_mapping and key_mapping[key] in api_data:
            value = api_data[key_mapping[key]]
            if key == "generated_summary":
                if not isinstance(value, str) or len(value.strip()) < 20:
                    logger.warning(f"[InputProvider] Invalid {key_mapping[key]} from API: {value}")
                    value = ""
                else:
                    # Check for key terms
                    if not any(term in value.lower() for term in ["deal", "pay tv", "rights", "count"]):
                        logger.warning(f"[InputProvider] generated_summary may lack specific details: {value[:100]}...")
            self.last_sources[key] = "API"
            return value
        if key == "sql_result":
            self.last_sources[key] = "Database"
            return None
        self.last_sources[key] = "Config"
        return self.config.get(key)
    
    def get_all(self, keys, prompt=None, session_id="default_session"):
        api_data = self.fetch_api_response()
        results = {}
        for key in keys:
            override_value = prompt if key == "user_intent" else None
            results[key] = self.get_value(
                key,
                override_value=override_value,
                api_data=api_data
            )
            logger.info(f"[InputProvider] {key} -> {results[key]} (source={self.last_sources[key]})")
        return results

    def get_sources(self):
        return self.last_sources

class SummaryValidator:
    def __init__(self, db_conn, config_path="config.yaml", api_url=None):
        self.db_conn = db_conn
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.input_provider = InputProvider(config=config, api_url=api_url)
        self.forward = ForwardValidator()
        self.num_questions = self.input_provider.get_value("num_questions") or 5
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
        questions = [q for q in answerability.keys() if q != "average_score"]
        factual_consistency = raw_results["forward_validation"]["factual_consistency"]
        return {
            "Forward Validation": {
                "Factual Consistency": {
                    "Result": self.status(factual_consistency["score"]),
                    "Confidence": round(factual_consistency["score"], 2),
                    "Claims": factual_consistency["details"]
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
            }
        }

    def run(self, user_prompt=None):
        keys = ["user_intent", "sql_result", "ground_truth", "questions", "sql_text", "generated_summary"]
        inputs = self.input_provider.get_all(keys, prompt=user_prompt)
        logger.info(f"[Validator] User intent: {inputs['user_intent']}")
        logger.info(f"[Validator] Questions: {inputs.get('questions')}")
        logger.info(f"[Validator] sql_text: {inputs.get('sql_text')}")
        logger.info(f"[Validator] generated_summary: {inputs.get('generated_summary')}")

        sql_text = inputs.get("sql_text", "")
        if not sql_text or not isinstance(sql_text, str):
            logger.error("[Validator] No valid sql_text provided; cannot fetch sql_result")
            raise ValueError("sql_text is required and must be a valid string")

        logger.info(f"[Validator] SQL query to execute:\n{sql_text}")
        try:
            sql_result = self.db_conn.execute_sql(sql_text, session_id="summary_validation")
            logger.info(f"[Validator] SQL query response: {json.dumps(sql_result[:3], indent=2, default=str)}")
        except Exception as e:
            logger.error(f"[Validator] Failed to execute SQL: {e}")
            raise

        generated_summary = inputs.get("generated_summary", "")
        if not generated_summary or not isinstance(generated_summary, str) or len(generated_summary.strip()) < 20:
            logger.warning(f"[Validator] Invalid generated_summary: {generated_summary}; using empty string")
            generated_summary = ""

        questions = inputs.get("questions", [])

        forward_results = {
            "factual_consistency": self.forward.factual_consistency(generated_summary, sql_result),
            "intent_alignment": self.forward.intent_alignment(generated_summary, inputs["user_intent"]),
            "answerability": self.forward.answerability(generated_summary, questions),
        }

        raw_results = {
            "data_source": "UserInput" if user_prompt else "API",
            "forward_validation": forward_results,
        }

        return self.interpret_results(raw_results)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    api_url = config.get("api_url")
    if not api_url:
        logger.error("[Main] No api_url in config.yaml; cannot fetch inputs")
        raise ValueError("api_url is required in config.yaml")
    db_conn = DatabaseConn()
    validator = SummaryValidator(db_conn=db_conn, config_path="config.yaml", api_url=api_url)
    try:
        result = validator.run(user_prompt=config.get("user_intent"))
    except Exception as e:
        logger.error(f"[Main] Validation failed: {e}")
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    output_path = Path("validation_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nValidation results saved to: {output_path.resolve()}")


