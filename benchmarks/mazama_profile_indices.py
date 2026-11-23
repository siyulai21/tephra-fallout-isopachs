"""Reproduce measured cryptotephra component indices, without creating fallout data."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'data/audits/Mazama'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(name):
    sources = ROOT/'data/sources/mazama'
    record = next(r for r in json.loads((sources/'source-manifest.json').read_text()) if r['file'] == name)
    if (sources/name).is_file():
        assert digest(sources/name) == record['sha256']
    return record['sha256']


def jensen_profiles():
    samples = list(csv.DictReader((AUDIT/'jensen2021-fine-profiles.csv').open()))
    windows = {
        'J2021-BB': (176, 182, [177, 180]),
        'J2021-SB-C2': (537, 539, [540]),
        'B2020-159': (545, 548, [548]),
        'J2021-TI': (345, 350, [346]),
    }
    results = {}
    for site, (lo, hi, identified_labels) in windows.items():
        part = [r for r in samples if r['record_id'] == site]
        labels = [int(r['depth_label_cm']) for r in part]
        assert labels == list(range(min(labels), max(labels)+1))
        values = {d: float(r['shards_per_cm3'])*float(r['width_cm']) for d,r in zip(labels,part)}
        gross = sum(values.values())
        results[site] = dict(source_table=part[0]['sheet'], observed_depth_labels_cm=labels,
            gross_window_shards_per_cm2=gross, alternate_depth_labels_inclusive_cm=[lo,hi],
            alternate_window_shards_per_cm2=sum(v for d,v in values.items() if lo <= d <= hi),
            selected_horizon_labels_cm=identified_labels,
            selected_slice_total_glass_per_cm2=sum(values[d] for d in identified_labels),
            largest_slice_concentration_shards_per_cm3=max(float(r['shards_per_cm3']) for r in part),
            nominal_maximum_window_response_if_T_at_least_t={
                str(t):gross/t for t in (.5,1.,5.,10.)},
            principal_original_fallout_bound_cm=None)
    return results


def northgrip():
    axes_file = AUDIT/'northgrip-axes.csv'
    with axes_file.open() as stream:
        lengths = np.array([float(r['long_axis_um']) for r in csv.DictReader(stream)])
    assert len(lengths) == 100 and np.mean(lengths) == 36.55
    r, w = 4.9, 1.
    area = (r*r*math.acos((r-w)/r) - (r-w)*math.sqrt(2*r*w-w*w))/2
    return dict(sample_name='NGRIP 1163.75', sample_depth_m=[1163.70,1163.75],
        measured_long_axes=len(lengths), measured_axis_min_mean_max_um=[lengths.min(),lengths.mean(),lengths.max()],
        source_files_sha256={n:source_digest(n) for n in ('northgrip-sizes.xlsx','northgrip-sup3.docx','northgrip-drill.html')},
        axis_transcription_sha256=digest(axes_file), recovered_shards=1200,
        major_analyses=18, trace_analyses=14, drill_diameter_cm=2*r, illustrated_edge_width_cm=w,
        full_core_max_area_cm2=math.pi*r*r, sampled_semicircle_edge_area_cm2=area,
        nominal_recovered_shards_per_cm2=1200/area,
        minimum_recovered_shards_per_cm2_if_full_disc=1200/(math.pi*r*r),
        measured_subset_box_volume_per_area_cm=float(np.sum((lengths/10000)**3)/area),
        all_shards_at_observed_80_um_max_box_volume_per_area_cm=1200*(80/10000)**3/area,
        effective_recovered_count_response_at_t_cm={str(t):1200/(area*t) for t in (.5,1.,5.,10.)},
        reading='Figure S1 shows a 1 cm edge cut through a semicircle of radius 4.9 cm. The area is half a circular segment. Scaling the count uses the illustrated cut for the resampled interval. It assumes the reported count corresponds to that sampled footprint. The diameter-only bound uses a complete circular slice.',
        volume_scope='Solid recovered-glass illustrations. The measured two-dimensional long axis must bound all three particle dimensions, and the all-particle illustration applies the measured maximum to the unmeasured particles. Missing grain sizes, vesicles, ice thinning, wind redistribution and the original bulk density limit transfer.',
        independent_visibility_control=dict(measured_visible_thickness_cm=.1,archive='GRIP, a different core, ice sample and tephra observed in another sampling campaign (Davies et al. 2024 section 2).',
            minimum_visible_response_needed_for_t_cm={str(t):.1/t for t in (.5,1.,5.,10.)},
            interpretation='Assuming a 0.1 cm visible-bed threshold would require a retained coherent fraction of at least 0.20 to exclude 0.5 cm. This control establishes that one distinct layer was visible. It does not calibrate that fraction for Mazama at NGRIP.'),
        principal_original_fallout_bound_cm=None)


def calculate():
    path = AUDIT/'cryptotephra-profile-samples.csv'
    rows = list(csv.DictReader(path.open()))
    result = {}
    for rid, background in [('B2020-110', 70), ('S2017-IR', 60)]:
        part = [r for r in rows if r['record_id'] == rid]
        n = np.array([int(r['shards']) for r in part])
        mass = np.array([float(r['wet_mass_g']) for r in part])
        dz = np.array([float(r['width_cm']) for r in part])
        reported = np.array([float(r['reported_shards_per_wet_g']) for r in part])
        assert len(part) == 10 and np.allclose(n/mass, reported, rtol=.0001)
        index = float(np.sum(n/mass*dz))
        # Bonferroni simultaneous Poisson intervals for the ten recovered counts.
        # This is a counting model, not a geological uncertainty distribution.
        tail = .05/(2*len(n))
        low = np.where(n == 0, 0., chi2.ppf(tail, 2*n)/2)
        high = chi2.ppf(1-tail, 2*(n+1))/2
        lo, hi = float(np.sum(low/mass*dz)), float(np.sum(high/mass*dz))
        result[rid] = dict(samples=[r['sample'] for r in part],
            window_m=[float(part[0]['mid_depth_m'])-.005, float(part[-1]['mid_depth_m'])+.005],
            units='shard cm / g wet sediment', raw_count_mass_index=index,
            wet_mass_sum_g=float(mass.sum()), raw_shards_sum=int(n.sum()),
            conditional_max_target_mass_g_per_cm2_if_full_one_cm2_column=float(mass.sum()),
            nominal_max_bulk_cm_if_full_capture_and_density_0_7=float(mass.sum()/.7),
            wet_mass_scope="Target dry mass cannot exceed all weighed wet material. An areal budget additionally needs these subsamples to exhaust the same known footprint at all depths and retain the complete event. The article gives an approximate 1 cm² inventory for both cores, with differing subsample weights and mixing. The 0.7 g/cm³ illustration is Young 1990 Table 1.1 upland standard, without evidence of a site-specific lower bound on original bulk density or capture.",
            sum_of_printed_rounded_concentrations=float(np.sum(reported*dz)),
            background_subtracted_index=float(np.sum(np.maximum(n/mass-background, 0)*dz)),
            background_shards_per_wet_g=background,
            background_scope='Reported approximate background used only as an alternative; not an independently estimated error or a required correction.',
            window_without_first_and_last_bin_index=float(np.sum((n/mass*dz)[1:-1])),
            counting_only_simultaneous_95_interval=[lo, hi],
            counting_assumptions='Poisson recovered counts with fixed reported masses, widths and representativeness; simultaneous Bonferroni interval. Does not include overdispersion, extraction loss, grain size, host density, sediment transport, attribution or unsampled tails.',
            response_upper_requirements=[dict(proposed_fallout_cm=t,
                nominal_index_per_cm_upper=index/t, counting_95_index_per_cm_upper=hi/t)
                for t in (.5, 1., 5., 10.)],
            response_meaning='For a total original thickness T>=t, a nonnegative target contribution within this window requires k_window<=J_upper/t. This is a requirement on an unknown response, not a thickness bound or inferred calibration.',
            principal_original_fallout_bound_cm=None)
    payload = dict(source_sha256=source_digest('spano2017.pdf'),
        sample_table_sha256=digest(path), script_sha256=digest(Path(__file__)),
        formula='J=sum((recovered_shards/wet_mass_g)*width_cm); process the fine bins once, never add overlapping 10 cm screening samples.',
        profiles=result,
        lake_superior_author_inventory=dict(approximate_fragments_per_cm2=[2000,4000],
            applies_to='Authors give an approximate range for both sites. Endpoints are not site assignments.',
            nominal_upper_solid_recovered_glass_cm=4000*(125/10000)**3,
            assumptions='Illustration only. A continuous 1 cm² footprint, 4000 counted fragments, and all three dimensions of every counted fragment at most 125 micrometres. The reported largest size and rounded inventory are not hard bounds. Unobserved glass, crystals, pores and redistribution remain unconstrained.'),
        jensen_source_sha256=source_digest('jensen2021-profiles.xlsx'),
        jensen_sample_table_sha256=digest(AUDIT/'jensen2021-fine-profiles.csv'),
        jensen_profiles=jensen_profiles(),
        jensen_scope='Gross glass concentration times contiguous 1 cm sample width. Blank recounts and overlapping 5 cm screening samples are excluded. Chemical identification of selected shards is not an attribution of all glass. Window endpoints are analyst choices, not bed contacts. Requirements on response use the gross component in that selected window and have no statistical confidence level.',
        other_recovered_components={
            'R2019-BCK284': dict(peak_shards_per_cm3=220, sample_width_cm=1,
                nominal_peak_slice_shards_per_cm2=220,
                scope='284–285 cm Mazama-associated slice only. Excludes older BCK-325; no conversion of sample width to ash thickness.'),
            'M2023-10G195': dict(window_cm=[190,195], concentration_upper_shards_per_dry_g=40,
                window_index_upper_shard_cm_per_dry_g=200,
                scope='Source-wide <40 shards/g ceiling times 5 cm, outward <=200. Finite-window component ceiling, not a whole-event bound; the source reports unresolved peaks and translocation.'),
            'D2024-NGRIP': northgrip(),
            'B2020-292': dict(core='GISP2', source_sha256=source_digest('zdanowicz1999.pdf'),
                four_positive_filters_m=[1325.11,1326.15], selected_filter_m=[1325.89,1326.15],
                analysed_shards_above_10_um=10, membrane_pore_um=.2,
                optical_shard_diameter_range_um='at most 1 to 20',
                electron_microscope_reference='Zdanowicz, Zielinski and Germani 1999 pp. 621–623, Figs. 1–3 and Table 1.',
                total_microparticle_counter_size_window_um=[.65,12],
                scope='Ten selected large grains verify attribution on one filter. The independent counter includes all insoluble particles within a different size window. Four positive filters span 104 cm of ice. The source gives neither a complete count of attributed grains in each filter nor the processed horizontal area. The sulphate peak estimates atmospheric sulphur, a different material.')},
        conclusion='No independently anchored count response, mass-transfer denominator or recognition floor was recovered. No numerical original-fallout limits are created by this calculation.')
    (AUDIT/'cryptotephra-profile-calculations.json').write_text(json.dumps(payload,indent=2)+'\n')
    for rid, row in result.items():
        print(rid, row['raw_count_mass_index'], row['counting_only_simultaneous_95_interval'])


if __name__ == '__main__':
    calculate()
