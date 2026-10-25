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
