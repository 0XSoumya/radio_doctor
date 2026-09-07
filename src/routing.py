"""Field routing logic for mapping clinical findings to template fields."""

import re
from typing import List, Dict, Optional
from src.templates import ParsedTemplate, TemplateField

# Standard anatomical and pathological associations
STANDARD_ROUTING_MAP = {
    r'\b(fracture|cortex|cortical|lesion|lytic|blastic|spondylosis|sclerosis|osteophyte|osteopenia|osteoporosis|bone|osseous|marrow|tuberosity|trochanter|ramus|acetabul|olecranon|patella|femur|tibia|fibula|humerus|radius|ulna|calcaneus|metatarsal|phalanx|phalanges)\b': ['BONES', 'OSSEOUS STRUCTURES', 'VERTEBRAE'],
    r'\b(joint|dislocation|subluxation|articular|cartilage|meniscus|meniscal|labrum|labral|narrowing|effusion|synovial|synovitis|osteoarthrosis|osteoarthritis|arthropathy|acromioclavicular|glenohumeral|sacroiliac)\b': ['JOINTS', 'GLENOHUMERAL JOINT', 'ACROMIOCLAVICULAR JOINT', 'CARTILAGE', 'LABRUM'],
    r'\b(tendon|supraspinatus|infraspinatus|subscapularis|teres|biceps|triceps|rotator cuff|achilles|plantar fascia|ligament|cruciate|collateral|atfl|cfl)\b': ['ROTATOR CUFF TENDONS AND MUSCLES', 'SUPRASPINATUS', 'INFRASPINATUS', 'SUBSCAPULARIS', 'TENDONS', 'LIGAMENTS', 'ACHILLES', 'SOFT TISSUES'],
    r'\b(soft tissue|swelling|edema|hematoma|collection|cyst|bursa|bursitis|lymph|adenopathy|mass)\b': ['SOFT TISSUES', 'PARASPINAL SOFT TISSUES', 'PERIARTICULAR SOFT TISSUE'],
    r'\b(disc|herniation|protrusion|extrusion|bulge|desiccation|annular|annulus|thecal|canal|stenosis|foraminal|foramina|neuroforaminal)\b': ['DISCS/DEGENERATIVE CHANGES', 'DISC SPACES', 'INTERVERTEBRAL DISC SPACES', 'VERTEBRAE'],
    r'\b(cord|myelomalacia|myelopathy|conus|cauda equina)\b': ['SPINAL CORD'],
    r'\b(lung|pulmonary|infiltrate|consolidation|opacity|atelectasis|pneumonia|nodule|bronch)\b': ['LUNGS', 'LUNGS/AIRWAYS'],
    r'\b(pleura|pleural|pneumothorax|effusion|thickening)\b': ['PLEURA', 'PLEURAL SPACES'],
    r'\b(heart|cardiac|cardiomegaly|aorta|aortic|mediastin|hilar|vascular)\b': ['HEART', 'CARDIOVASCULAR', 'MEDIASTINUM/HILA'],
    r'\b(brain|infarct|hemorrhage|ischemi|leuko|atrophy|ventricle|hydrocephalus|gray-white|mass effect|midline shift)\b': ['BRAIN', 'VENTRICLES'],
}


def find_best_field_for_finding(
    finding_text: str,
    target_template: ParsedTemplate
) -> str:
    """Find the best matching field in target_template for a finding."""
    valid_fields = [f.name for f in target_template.fields if not f.is_header]
    valid_fields_upper = {f.upper(): f for f in valid_fields}

    # 1. Exact or partial match with a field name
    text_lower = finding_text.lower()
    for fname in valid_fields:
        if fname.lower() in text_lower or text_lower in fname.lower():
            return fname

    # 2. Rule-based anatomical keyword matching
    for pattern, preferred_fields in STANDARD_ROUTING_MAP.items():
        if re.search(pattern, text_lower):
            for pf in preferred_fields:
                if pf in valid_fields_upper:
                    return valid_fields_upper[pf]

    # 3. Fall back to OTHER FINDINGS if present or permitted, else first non-header field
    if "OTHER FINDINGS" in valid_fields_upper:
        return valid_fields_upper["OTHER FINDINGS"]

    return valid_fields[0] if valid_fields else "OTHER FINDINGS"
