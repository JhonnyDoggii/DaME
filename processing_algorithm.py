# ======================== IMPORTS ==================================
from qgis.PyQt.QtCore import QCoreApplication
from PyQt5.QtCore import QDate, QDateTime
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterString,
    QgsProcessingParameterFileDestination,
    QgsWkbTypes,
    QgsRasterBandStats,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRectangle
    
)
import json, xml.etree.ElementTree as ET, time, os, re
import requests

# ======================== LLM HELPERS ==============================
def _post_with_retry(url, headers, payload, feedback, max_retries=5, backoff_base=2, timeout=60):
    """
    Generic POST with exponential backoff for 429 errors.
    Returns response JSON or raises the last exception.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = None
            try:
                status = r.status_code
            except Exception:
                status = None

            if status == 429:
                wait = backoff_base * (attempt + 1)
                feedback.pushInfo(f"Rate limited (429). Retrying in {wait} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            else:
                # non-retryable HTTP error
                feedback.pushInfo(f"HTTP error: {e}")
                raise
        except Exception as e:
            last_exc = e
            feedback.pushInfo(f"Request error: {e}")
            # For network errors, also back off a bit
            wait = backoff_base * (attempt + 1)
            time.sleep(wait)
            continue
    # exhausted retries
    feedback.pushInfo("Exhausted retries for LLM request.")
    raise last_exc if last_exc is not None else RuntimeError("Unknown LLM request failure")


def call_llm_field_translation(fields, feedback,
                               base_url="https://api.llm7.io/v1",
                               model="gpt-4o-mini-2024-07-18"):
    """
    Token-safe translation of field names.
    Returns mapping {original: translated} (fallback identity mapping on error).
    """
    if not fields:
        return {}
    feedback.pushInfo("Translating field names (token-safe)...")

    prompt = (
        "Translate ONLY the following field names into English. "
        "Keep proper nouns unchanged. "
        "Return exactly one JSON object mapping original->translation. No explanations.\n\n"
        f"{json.dumps(fields, ensure_ascii=False)}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300
    }

    try:
        resp = _post_with_retry(f"{base_url}/chat/completions",
                                headers={"Content-Type": "application/json"},
                                payload=payload,
                                feedback=feedback,
                                max_retries=5, backoff_base=2, timeout=60)
        txt = resp["choices"][0]["message"]["content"].strip()
        j = re.search(r"\{.*\}", txt, re.S)
        if not j:
            raise ValueError("No JSON block in LLM response")
        return json.loads(j.group())
    except Exception as e:
        feedback.pushInfo(f"Field translation failed, using identity mapping: {e}")
        return {f: f for f in fields}


def call_llm_generate_intro(layer_name, filename, feedback,
                            base_url="https://api.llm7.io/v1",
                            model="gpt-4o-mini-2024-07-18"):
    """
    Generate a short (1-2 sentence) polished intro paragraph mentioning the
    dataset name only (NO filename included per your request).
    Uses retry logic to handle 429s.
    Returns a single-line string or None on failure.
    """
    feedback.pushInfo("Generating LLM-enhanced intro (token-safe)...")

    prompt = (
        "Write a short (1–2 sentence) professional introductory paragraph "
        "for a geospatial dataset. You may mention the dataset name. "
        "Do NOT mention filenames. "
        "Do NOT invent numbers, statistics, or details about fields. "
        "Do NOT include bullet lists or questions. "
        "Return only the paragraph.\n\n"
        f"Dataset name: {layer_name}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.35,
        "max_tokens": 300
    }

    try:
        resp = _post_with_retry(f"{base_url}/chat/completions",
                                headers={"Content-Type": "application/json"},
                                payload=payload,
                                feedback=feedback,
                                max_retries=5, backoff_base=2, timeout=60)
        text = resp["choices"][0]["message"]["content"].strip()
        if not text:
            return None
        single = " ".join(text.split())
        if re.search(r"(please provide|could you|i cannot|i need)", single, re.I):
            return None
        return single
    except Exception as e:
        feedback.pushInfo(f"Intro generation failed after retries: {e}")
        return None


def call_llm_refine_overview(overview, feedback,
                             base_url="https://api.llm7.io/v1",
                             model="gpt-4o-mini-2024-07-18"):
    """
    Backward-compatible refine function (sentence-level, hallucination-proof).
    Not used for rewriting attributes in new flow, but kept available.
    """
    feedback.pushInfo("Refining overview (flowing paragraph, hallucination-proof)...")
    if not overview or not isinstance(overview, str):
        return overview

    sentences = [s.strip() for s in overview.split(". ") if s is not None and s.strip() != ""]
    refined = []
    unsafe_pattern = re.compile(r"(The field\s+'[^']+'|'\w+'|:|\d)")

    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) > 200 or unsafe_pattern.search(s_clean):
            refined.append(s_clean)
            continue

        prompt = (
            "Rewrite this sentence to improve clarity and flow in English. "
            "Do NOT introduce any information that is not already present. "
            "Do NOT ask questions, add explanations, or provide meta-comments. "
            "Keep the rewritten result as a single sentence only. "
            "Return ONLY the rewritten sentence.\n\n"
            f"{s_clean}"
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300
        }

        try:
            resp = _post_with_retry(f"{base_url}/chat/completions",
                                    headers={"Content-Type": "application/json"},
                                    payload=payload,
                                    feedback=feedback,
                                    max_retries=3, backoff_base=2, timeout=60)
            out = resp["choices"][0]["message"]["content"].strip()
            if not out or len(out) < 3 or re.search(r"(provide the short sentence|please provide)", out, re.I):
                refined.append(s_clean)
            else:
                out_one_line = " ".join(out.splitlines()).strip()
                refined.append(out_one_line)
        except Exception as e:
            feedback.pushInfo(f"LLM refine error (keeping original): {e}")
            refined.append(s_clean)

    normalized = []
    for s in refined:
        s = s.strip()
        if not s:
            continue
        if s[-1] not in ".!?":
            s = s + "."
        normalized.append(s)

    paragraph = " ".join(normalized).strip()
    paragraph = re.sub(r"\s+", " ", paragraph)
    return paragraph

# ================= XML / DICT HELPERS ================================

def dict_to_xml(tag, d):
    elem = ET.Element(tag)
    if isinstance(d, dict):
        for k, v in d.items():
            child = ET.SubElement(elem, k)
            if isinstance(v, dict):
                sub = dict_to_xml("item", v)
                for c in list(sub):
                    child.append(c)
            elif isinstance(v, list):
                for item in v:
                    it = ET.SubElement(child, "item")
                    if isinstance(item, dict):
                        for kk, vv in item.items():
                            subchild = ET.SubElement(it, kk)
                            subchild.text = str(vv)
                    else:
                        it.text = str(item)
            else:
                child.text = "" if v is None else str(v)
    else:
        elem.text = str(d)
    return elem

def xml_to_dict(elem):
    children = list(elem)
    if not children:
        return elem.text
    d = {}
    for child in children:
        d[child.tag] = xml_to_dict(child)
    return d

# ================= EXTRACTION HELPERS (VECTOR / RASTER) ============
def _safe_values_list(values):
    out = []
    for v in values:
        if v in [None, "", "NaN", "NULL"]:
            continue

        if isinstance(v, QDate):
            out.append(v.toString("yyyy-MM-dd"))
        elif isinstance(v, QDateTime):
            out.append(v.toString("yyyy-MM-ddTHH:mm:ss"))
        else:
            out.append(v)

    return out

def build_field_description(f, vals):
    vals_clean = _safe_values_list(vals)
    if not vals_clean:
        return None

    if isinstance(vals_clean[0], (int, float)):
        vmin, vmax = min(vals_clean), max(vals_clean)
        n_unique = len(set(vals_clean))
        return f"The field '{f}' ranges from {vmin} to {vmax} (unique values: {n_unique})"
    else:
        uniq = sorted(set(vals_clean))
        n_unique = len(uniq)
        examples = ", ".join(map(str, uniq[:10]))
        return f"The field '{f}' contains {n_unique} unique categories such as: {examples}"

# ======================= MAIN QGIS ALGORITHM =======================

class DataAwareMetadataEnricher(QgsProcessingAlgorithm):

    INPUT_VECTOR = 'INPUT_VECTOR'
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_METADATA = 'INPUT_METADATA'
    OUTPUT_METADATA = 'OUTPUT_METADATA'
    FIELDS = 'FIELDS'
    SAMPLE_SIZE = 'SAMPLE_SIZE'
    USE_LLM = 'USE_LLM'
    LLM_MODEL = 'LLM_MODEL'
    EXTRACT_RASTER_STATS = 'EXTRACT_RASTER_STATS'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return DataAwareMetadataEnricher()

    def name(self):
        return 'data_aware_metadata_enricher'

    def displayName(self):
        return self.tr('Data-aware Metadata Enricher')

    def group(self):
        return self.tr('Metadata tools')

    def groupId(self):
        return 'metadata_tools'

    def shortHelpString(self):
        return self.tr('Scan a vector or raster layer and enrich an existing metadata file (JSON or XML).')

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.INPUT_VECTOR,
            self.tr('Input vector layer (leave empty if using raster)'),
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_RASTER,
            self.tr('Input raster layer (leave empty if using vector)'),
            optional=True
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.INPUT_METADATA,
            self.tr('Existing metadata file (JSON or XML)'),
            optional=False,
        ))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_METADATA,
            self.tr('Output enriched metadata file (JSON or XML)'),
            fileFilter='JSON files (*.json);;XML files (*.xml)',
        ))
        self.addParameter(QgsProcessingParameterField(
            self.FIELDS,
            self.tr('Fields to use for enrichment (vector only)'),
            parentLayerParameterName=self.INPUT_VECTOR,
            optional=True,
            allowMultiple=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.SAMPLE_SIZE,
            self.tr('Max sample size for attribute scanning (0 = no limit)'),
            minValue=0,
            defaultValue=500,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_LLM,
            self.tr('Use LLM to translate field names and enrich descriptions'),
            defaultValue=False,
        ))
        self.addParameter(QgsProcessingParameterString(
            self.LLM_MODEL,
            self.tr('LLM model'),
            defaultValue="gpt-4o-mini-2024-07-18",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.EXTRACT_RASTER_STATS,
            self.tr('Extract raster statistics (min, max, mean, stddev, median)'),
            defaultValue=True,
        ))

        # -------------------------- PROCESS --------------------------------
    def processAlgorithm(self, parameters, context, feedback):

        vector_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        input_meta_path = self.parameterAsFile(parameters, self.INPUT_METADATA, context)
        output_meta_path = self.parameterAsFile(parameters, self.OUTPUT_METADATA, context)
        fields = self.parameterAsFields(parameters, self.FIELDS, context) or []
        sample_size = int(self.parameterAsDouble(parameters, self.SAMPLE_SIZE, context))
        use_llm = self.parameterAsBool(parameters, self.USE_LLM, context)
        llm_model = self.parameterAsString(parameters, self.LLM_MODEL, context)
        extract_raster_stats = self.parameterAsBool(parameters, self.EXTRACT_RASTER_STATS, context)

        feedback.pushInfo("Starting metadata enrichment...")

        # ---------------- Read existing metadata ----------------
        existing_meta = None
        existing_type = None
        if input_meta_path:
            try:
                if input_meta_path.lower().endswith(".json"):
                    with open(input_meta_path, "r", encoding="utf-8") as f:
                        existing_meta = json.load(f)
                    existing_type = "json"
                else:
                    existing_meta = ET.parse(input_meta_path)
                    existing_type = "xml"
            except Exception as e:
                feedback.pushInfo(f"Could not read existing metadata: {e}")

        # ---------------- Validate input ----------------
        if not input_meta_path:
            raise QgsProcessingException(
                "An existing metadata file is required. Please provide a JSON or XML metadata file."
            )
        if vector_layer and raster_layer:
            raise QgsProcessingException(
                "Please provide only ONE input layer: either a vector or a raster, not both."
            )
        if not vector_layer and not raster_layer:
            raise QgsProcessingException(
                "Please provide at least ONE input layer: either a vector or a raster."
            )
            return {self.OUTPUT_METADATA: output_meta_path}


        # ---------------- Extract metadata ----------------
        if vector_layer:
            extracted = self.extract_vector_metadata(vector_layer, fields, sample_size, feedback)
        else:
            extracted = self.extract_raster_metadata(raster_layer, sample_size, feedback, extract_raster_stats)

        # ---------------- Merge metadata ----------------
        merged = self.merge_metadata(existing_meta, existing_type, extracted, feedback)

        # ---------------- LLM ENHANCEMENT -----------------------------
        
        if use_llm and isinstance(merged, dict) and "descriptive_overview" in merged:
            overview = merged["descriptive_overview"]

            # ---- Translate field names ----
            fields_in_text = re.findall(r"The field '([^']+)'", overview)
            fields_in_text = list(dict.fromkeys(fields_in_text))
            if fields_in_text:
                mapping = call_llm_field_translation(fields_in_text, feedback, model=llm_model)
                for orig, trans in mapping.items():
                    if trans and trans.lower() != orig.lower():
                        overview = overview.replace(f"'{orig}'", f"'{orig} ({trans})'")
                merged["descriptive_overview"] = overview

            # ---- Enhance flowing paragraph with LLM intro ----
            intro = call_llm_generate_intro(merged.get("layer_name", "dataset"), None, feedback, model=llm_model)
            refined_para = call_llm_refine_overview(overview, feedback, model=llm_model)
            merged["descriptive_overview"] = (intro + " " + refined_para) if intro else refined_para

        # ---------------------------- OUTPUT -------------------------------
        ext = os.path.splitext(output_meta_path)[1].lower()
        if ext not in [".json", ".xml"]:
            ext = ".json"
            output_meta_path += ".json"

        try:
            if ext == ".json":
                with open(output_meta_path, "w", encoding="utf-8") as f:
                    json.dump(merged if isinstance(merged, dict) else {}, f,
                              indent=2, ensure_ascii=False)
            else:
                root = dict_to_xml("metadata", merged)
                ET.ElementTree(root).write(output_meta_path, encoding="utf-8", xml_declaration=True)
            feedback.pushInfo(f"Metadata written to: {output_meta_path}")
        except Exception as e:
            feedback.reportError(f"Writing output failed: {e}")

        return {self.OUTPUT_METADATA: output_meta_path}

   
    # ------------------- Metadata Extraction ---------------------------

    def extract_vector_metadata(self, layer, fields, sample_size, feedback):
        meta = {}
        meta['layer_name'] = layer.name()
        meta['source'] = layer.source()
        meta['crs'] = layer.crs().authid() if layer.crs().isValid() else None

        extent = layer.extent()

        # ---- Reproject bounding box only → EPSG:4326 ----
        src_crs = layer.crs()
        dst_crs = QgsCoordinateReferenceSystem("EPSG:4326")

        feedback.pushInfo(f" Source CRS: {src_crs.authid()} | Geographic: {src_crs.isGeographic()}")

        if src_crs.isValid() and src_crs.authid() != "EPSG:4326":
            try:
                transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
                # Transform all four corners manually (bulletproof)
                ll = transform.transform(extent.xMinimum(), extent.yMinimum())
                lr = transform.transform(extent.xMaximum(), extent.yMinimum())
                ul = transform.transform(extent.xMinimum(), extent.yMaximum())
                ur = transform.transform(extent.xMaximum(), extent.yMaximum())

                xs = [ll.x(), lr.x(), ul.x(), ur.x()]
                ys = [ll.y(), lr.y(), ul.y(), ur.y()]

                extent = QgsRectangle(min(xs), min(ys), max(xs), max(ys))

            except Exception as e:
                feedback.pushInfo(f"BBOX reprojection failed: {e}")

        meta['spatial'] = dict(
            xmin=extent.xMinimum(),
            ymin=extent.yMinimum(),
            xmax=extent.xMaximum(),
            ymax=extent.yMaximum(),
        )

        meta['geometry_type'] = QgsWkbTypes.displayString(layer.wkbType())
        meta['feature_count'] = layer.featureCount()

        fields_to_scan = fields if fields else [f.name() for f in layer.fields()]
        values = {f: [] for f in fields_to_scan}
        sampled = 0

        for feat in layer.getFeatures():
            if sample_size > 0 and sampled >= sample_size:
                break
            for f in fields_to_scan:
                try:
                    values[f].append(feat[f])
                except Exception:
                    pass
            sampled += 1

        summary = {}
        descriptions = []

        for f in fields_to_scan:
            vals = values[f]
            desc = build_field_description(f, vals)
            if desc:
                descriptions.append(desc)

            vals_clean = _safe_values_list(vals)
            if not vals_clean:
                continue

            if isinstance(vals_clean[0], (int, float)):
                vmin, vmax = min(vals_clean), max(vals_clean)
                n_unique = len(set(vals_clean))
                mean_v = sum(vals_clean)/len(vals_clean)
                summary[f] = dict(
                    type="numeric",
                    range=[vmin, vmax],
                    unique_values=n_unique,
                    mean=mean_v
                )
            else:
                uniq = sorted(set(vals_clean))
                summary[f] = dict(
                    type="categorical",
                    unique_values=len(uniq),
                    values=uniq
                )

        meta["fields_summary"] = summary
        meta["descriptive_overview"] = "This dataset includes several important attributes: " + " ".join(descriptions)
        meta["extracted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return {"type": "vector", "summary": meta}

    # ---------------- Extract Raster Metadata ----------------
    def extract_raster_metadata(self, layer, sample_size, feedback, extract_raster_stats=True):
        meta = {}
        meta["layer_name"] = layer.name()
        meta["source"] = layer.source()
        meta["crs"] = layer.crs().authid() if layer.crs().isValid() else None

        extent = layer.extent()

        # ---- Reproject bounding box only → EPSG:4326 ----
        src_crs = layer.crs()
        dst_crs = QgsCoordinateReferenceSystem("EPSG:4326")

        if src_crs.isValid() and not src_crs.isGeographic():
            try:
                transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
                extent = transform.transformBoundingBox(extent)
            except Exception as e:
                feedback.pushInfo(f" BBOX reprojection failed: {e}")

        meta["spatial"] = dict(
            xmin=extent.xMinimum(),
            ymin=extent.yMinimum(),
            xmax=extent.xMaximum(),
            ymax=extent.yMaximum(),
        )

        if extract_raster_stats:
            provider = layer.dataProvider()
            band_count = layer.bandCount()
            stats = {}

            for band in range(1, band_count + 1):
                try:
                    s = provider.bandStatistics(band, QgsRasterBandStats.All)
                    stats[f"band_{band}"] = dict(
                        min=s.minimumValue,
                        max=s.maximumValue,
                        mean=s.mean,
                        stddev=s.stdDev,
                        median_approx=None,
                    )
                except Exception as e:
                    feedback.pushInfo(f" Band {band} stats failed: {e}")

            meta["raster_band_statistics"] = stats

        meta["extracted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["descriptive_overview"] = f"This raster dataset '{layer.name()}' includes {layer.bandCount()} bands."

        return {"type": "raster", "summary": meta}

    # ------------------------ Merge Logic ------------------------------
 
    def merge_metadata(self, existing, existing_type, extracted, feedback):
        extracted = extracted.get("summary", extracted)
        prov = dict(last_extracted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    tool="qgis-data-aware-metadata-enricher/0.7")
        if existing is None:
            merged = extracted.copy()
            merged["prov"] = prov
            return merged
        if existing_type == "json":
            merged = existing.copy()
            merged.update(extracted)
            merged["prov"] = prov
            return merged
        # XML case
        try:
            xml_dict = xml_to_dict(existing.getroot())
            xml_dict.update(extracted)
            xml_dict["prov"] = prov
            return xml_dict
        except Exception as e:
            feedback.pushInfo(f" XML merge failed: {e}")
            merged = extracted.copy()
            merged["prov"] = prov
            return merged
