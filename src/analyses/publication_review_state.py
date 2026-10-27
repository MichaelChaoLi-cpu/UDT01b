"""Refresh live companion prose from hash-bound output acceptance receipts.

Creation-time validation and completion receipts are historical and never edited here.
Changed outputs cannot inherit an earlier acceptance unless its hashes still match.
"""
from pathlib import Path
import hashlib
import html
import json
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_review_page(root, slug):
    root = Path(root)
    directory = root / 'data/exp/figure-table-generation/three_ward_transport_v1'
    special = slug == 'study_area_and_urban_flood_computation'
    receipt_path = directory / (slug + '_acceptance.json')
    page = directory / (slug + '_review.html')
    methods = directory / (slug + '_methods.md')
    if special:
        receipt_path = directory / slug / 'acceptance.json'
        page = directory / slug / 'review.html'
        methods = directory / slug / 'methods.md'
    figure = root / f'data/results/figures/Figure_{slug}.png'
    workbook = root / f'data/results/tables/Table_{slug}.xlsx'
    output = figure if figure.is_file() else workbook
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    accepted = receipt.get('status') in ('human_accepted', 'done')
    png_key = 'png_sha256' if 'png_sha256' in receipt else 'output_sha256'
    if figure.exists():
        accepted = accepted and digest(figure) == receipt.get(png_key)
    else:
        png = workbook.with_suffix('.png')
        accepted = (accepted and digest(workbook) == receipt.get('workbook_sha256')
                    and png.exists() and digest(png) == receipt.get('png_sha256'))
    status = ('Accepted output; current artifact hashes match the recorded review. Decision: '
              + receipt.get('decision', '') + '.') if accepted else (
              'Generated output; individual visual acceptance remains pending. '
              'A layout-plan approval does not constitute output acceptance.')
    # Remove obsolete live-state sentences; preserve units, captions and scientific scope.
    obsolete = [
        'Pending review. ', 'Pending output review.',
        'New assembly pending individual visual review. ',
        'Five-panel revision awaiting visual acceptance. ',
        'Six-panel assembly awaiting visual acceptance. ',
        'Staggered six-panel trial; awaiting visual review. ',
        'This revised figure awaits visual acceptance; later outputs remain paused.',
        'The generated figure remains pending individual visual acceptance; later planned outputs remain paused.',
        'The current revision awaits visual acceptance; later planned outputs remain paused.',
        'The revised output remains pending human review.',
        'This new assembly awaits individual human review.',
        'It remains pending human review; no subsequent table is automatically generated.',
        'The workbook contains the complete source trace and state-lineage register. The earlier cross-event figure revision remains pending final acceptance; the accepted resident, adverse-change and traffic images are preserved. This table awaits its own review.',
        'The trial stays pending human visual acceptance, with no new simulation or route computation.',
        'English 450 dpi PNG is rendered separately for visual review; both table acceptance statuses remain pending.',
        'Four-panel draft · English · 450 dpi · Existing model inputs only',
    ]
    for path in [page, methods]:
        if not path.exists():
            continue
        text = path.read_text()
        for old in obsolete:
            text = text.replace(old, '')
        if path == page:
            text = re.sub(r'<p id="publication-review-state"[^>]*>.*?</p>', '', text, flags=re.S)
            banner = '<p id="publication-review-state" style="padding:12px;background:#e3eef3">' + html.escape(status) + '</p>'
            text, count = re.subn(r'(</h1>)', lambda m: m.group(1) + banner, text, count=1)
            assert count == 1, page
        else:
            text = re.sub(r'\n<!-- live-review-state -->.*?<!-- /live-review-state -->\n?', '\n', text, flags=re.S)
            text += '\n<!-- live-review-state -->\n' + status + '\nCreation-time validation receipts retain their original generation status.\n<!-- /live-review-state -->\n'
        path.write_text(text)
    state = {
        'slug': slug, 'status': 'human_accepted' if accepted else 'pending_human_acceptance',
        'acceptance_decision': receipt.get('decision') if accepted else None,
        'acceptance_hashes_match': bool(accepted),
        'artifact_sha256': digest(output),
        'creation_time_receipts_preserved': True,
    }
    (directory / (slug + '_current_review_state.json')).write_text(json.dumps(state, indent=2) + '\n')
    if workbook.exists():
        refresh_table_roles(root)
    return state


def refresh_table_roles(root):
    """Keep the current role register separate from generation-time receipts."""
    directory = root / 'data/exp/figure-table-generation/three_ward_transport_v1'
    groups = {
        'main_text': [('experimental_design', 'Experimental Design'),
                      ('deployment_assumptions_and_receiving_limits', 'Deployment Assumptions and Receiving Limits'),
                      ('physical_road_and_resident_comparison', 'Resident Access Benefits')],
        'supplementary': [('adverse_transfer_and_coverage', 'Adverse Transfer and Coverage'),
                          ('vehicle_request_and_common_cohort_cost', 'Vehicle Request and Common-Cohort Cost'),
                          ('scenario_coverage_and_evidence_limits', 'Scenario Coverage and Evidence Limits')],
    }
    roles = {'decision': 'MILI-D-20261025-009', 'resident_stable_identifier': 'Physical Road and Resident Comparison',
             'coverage_workbook_shape': [9, 8], 'coverage_preview_shape': [9, 6]}
    for group, entries in groups.items():
        roles[group] = []
        for slug, title in entries:
            receipt = json.loads((directory / (slug + '_acceptance.json')).read_text())
            workbook = root / f'data/results/tables/Table_{slug}.xlsx'
            accepted = (receipt.get('status') == 'human_accepted'
                        and digest(workbook) == receipt.get('workbook_sha256')
                        and digest(workbook.with_suffix('.png')) == receipt.get('png_sha256'))
            roles[group].append({'title': title, 'status': 'human_accepted' if accepted else 'pending'})
    (directory / 'table_publication_roles.json').write_text(json.dumps(roles, indent=2) + '\n')
