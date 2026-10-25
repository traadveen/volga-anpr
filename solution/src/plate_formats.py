"""Current Russian collection formats; frozen legacy formats remain unchanged."""
import math

ALPHABET = 'ABEKMHOPCTYX'


def position_mask(plate_type, length):
    if plate_type in ('type1','type1a') and length in (8,9):return 'LDDDLL'+'D'*(length-6)
    if plate_type=='type1b' and length in (7,8):return 'LLDDD'+'D'*(length-5)
    raise ValueError('Unsupported plate type or text length')


def valid_plate(text,plate_type,partial=False):
    if not isinstance(text,str):return False
    try:mask=position_mask(plate_type,len(text))
    except ValueError:return False
    return all(ch in ((ALPHABET if slot=='L' else '0123456789')+('#' if partial else '')) for ch,slot in zip(text,mask))


def yellow_postprocess(raw,probabilities):
    """Mask uncertain/illegal positions; never guess, pad or substitute characters."""
    if not isinstance(raw,str):return '', 'missing_raw_text'
    try:mask=position_mask('type1b',len(raw))
    except ValueError:return '', 'unsupported_raw_length'
    if not isinstance(probabilities,(list,tuple)) or len(probabilities)!=len(raw):return '', 'probability_length_mismatch'
    values=[]
    for probability in probabilities:
        if isinstance(probability,bool) or not isinstance(probability,(int,float)) or not math.isfinite(probability) or not 0<=probability<=1:
            return '', 'invalid_character_probability'
        values.append(probability)
    text=''.join(ch if ch in (ALPHABET if slot=='L' else '0123456789') and confidence>=(.7 if slot=='L' else .5)
                 else '#' for ch,slot,confidence in zip(raw,mask,values))
    return (text,'typed_confidence_mask') if set(text)!={'#'} else ('','all_positions_unknown')


def apply_russian_profile(result):
    """Use observed detector type and OCR evidence only. Input results stay intact."""
    if not any(d.get('plate_type')=='type1b' for d in result['detections']) and not any(p.get('plate_type')=='type1b' for p in result['predictions']):return result
    records=[];replacements=[];diagnostics=[]
    for source in result['detections']:
        row=dict(source)
        if row.get('plate_type')=='type1b':
            final,reason=yellow_postprocess(row.get('raw_ocr'),row.get('char_probs'))
            confidence=row.get('ocr_confidence')
            if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:
                final='';reason='invalid_sequence_confidence'
            row.update(frozen_final_ocr=source.get('final_ocr'),final_ocr=final,
                       typed_postprocessed_ocr=final,recognition_profile='russian_v1',typed_postprocess_reason=reason)
            diagnostics.append({'detection_index':row['detection_index'],'reason':reason,'emitted':bool(final)})
            if final:
                replacements.append({'image':row.get('image',result['image']),'plate_num':final,
                    'plate_type':'type1b','confidence':confidence,'detection_index':row['detection_index']})
        records.append(row)
    predictions=[dict(p) for p in result['predictions'] if p['plate_type']!='type1b']+replacements
    predictions.sort(key=lambda p:p['detection_index'])
    return {**result,'detections':records,'predictions':predictions,'recognition_profile':'russian_v1',
            'type1b_postprocess_diagnostics':diagnostics}


FORMAT_PROFILE = 'format_consistency_v1'


def _complete_white_text(row):
    """Existing character thresholds, strict complete white format, no substitutions."""
    raw = row.get('raw_ocr')
    if not valid_plate(raw, 'type1'):
        return None
    probabilities = row.get('char_probs')
    confidence = row.get('ocr_confidence')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != len(raw):
        return None
    for char, probability in zip(raw, probabilities):
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            return None
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            return None
        # The existing OCR mask uses the same thresholds. No new tuning.
        if probability < (.5 if char in '0123456789' else .7):
            return None
    return raw


def _white_single_line_evidence(row):
    """A detector type may be contradicted only by complete observed evidence."""
    if row.get('plate_type') != 'type1b' or row.get('status') != 'ocr':
        return None
    raw = _complete_white_text(row)
    if raw is None:
        return None
    try:
        from src.geometry_f import validate_quad
        quad = validate_quad(row.get('observed_quad'))
        width = (math.dist(quad[0], quad[1]) + math.dist(quad[3], quad[2])) / 2
        height = (math.dist(quad[0], quad[3]) + math.dist(quad[1], quad[2])) / 2
        ratio = width / height
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    # Same boundary as RouterF. A square or missing quad is not sufficient.
    if not math.isfinite(ratio) or ratio <= 3.:
        return None
    return raw, ratio


def apply_format_consistent_profile(result):
    """Preserve the yellow profile unless full white text and long geometry agree.

    This changes neither OCR symbols nor detector_class. The final plate_type
    records the evidence-based correction; the original classification stays
    available in diagnostics. No labels, filenames or image identities are used.
    """
    baseline = apply_russian_profile(result)
    records = []
    predictions = {row['detection_index']: dict(row) for row in baseline['predictions']}
    source_by_index = {row['detection_index']: row for row in result['detections']}
    corrections = []
    auxiliary = set(result.get('ordinary_reference_detection_indices', []))
    auxiliary_decisions = []
    for source in baseline['detections']:
        row = dict(source)
        evidence = _white_single_line_evidence(source_by_index[row['detection_index']])
        if evidence is not None:
            text, ratio = evidence
            reason = 'complete_white_format_and_long_observed_quad'
            row.update(
                original_plate_type=row['plate_type'], plate_type='type1',
                baseline_postprocess_reason=row.get('typed_postprocess_reason'),
                final_ocr=text, typed_postprocessed_ocr=text,
                recognition_profile=FORMAT_PROFILE, typed_postprocess_reason=reason,
                type_correction_evidence={'observed_quad_ratio': ratio, 'raw_format': 'LDDDLLRR_or_RRR'},
            )
            predictions[row['detection_index']] = dict(
                image=row.get('image', result['image']), plate_num=text,
                plate_type='type1', confidence=row['ocr_confidence'],
                detection_index=row['detection_index'],
            )
            corrections.append(dict(
                detection_index=row['detection_index'], previous_type='type1b',
                final_type='type1', reason=reason, observed_quad_ratio=ratio,
            ))
        if row['detection_index'] in auxiliary:
            complete = _complete_white_text(source_by_index[row['detection_index']])
            accepted = complete is not None and row.get('final_ocr') == complete and row.get('plate_type') == 'type1'
            row['ordinary_reference_accepted'] = accepted
            auxiliary_decisions.append(dict(detection_index=row['detection_index'], accepted=accepted))
            if not accepted:
                predictions.pop(row['detection_index'], None)
                row.update(final_ocr='', ordinary_reference_reason='incomplete_or_uncertain_white_format')
        records.append(row)
    diagnostics = []
    corrected_indices = {row['detection_index'] for row in corrections}
    for source in baseline.get('type1b_postprocess_diagnostics', []):
        row = dict(source)
        if row['detection_index'] in corrected_indices:
            row.update(baseline_reason=row['reason'],
                       reason='complete_white_format_and_long_observed_quad', emitted=True)
        diagnostics.append(row)
    return {
        **baseline, 'predictions': [predictions[index] for index in sorted(predictions)],
        'detections': records, 'recognition_profile': FORMAT_PROFILE,
        'type1b_postprocess_diagnostics': diagnostics,
        'format_type_corrections': corrections,
        'ordinary_reference_decisions': auxiliary_decisions,
    }
