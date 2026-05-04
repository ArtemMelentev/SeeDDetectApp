import json

from segment_seeds_scan import analyze_image


def run_analysis_json(input_path, output_dir, release_mode=False):
    result = analyze_image(input_path, output_dir, release_mode=bool(release_mode))
    return json.dumps(result, ensure_ascii=False)
