import json

from segment_seeds_scan import analyze_image


def run_analysis_json(input_path, output_dir):
    result = analyze_image(input_path, output_dir)
    return json.dumps(result, ensure_ascii=False)
