"""Experimental spatial support only; this module does not classify plate types."""
import math


def rectangle(value):
    if len(value)!=4 or not all(math.isfinite(float(v)) for v in value):
        raise ValueError('Expected four finite bbox coordinates')
    x1,y1,x2,y2=map(float,value)
    if x2<=x1 or y2<=y1:raise ValueError('Degenerate bbox')
    return x1,y1,x2,y2


def vehicle_support(plate,vehicles,min_intersection=.8):
    """Require the plate center and >=80% of its box inside one vehicle box.

    Inputs contain observed boxes only. No plate text, type or GT is consulted.
    """
    if not 0<min_intersection<=1:raise ValueError('Invalid intersection fraction')
    x1,y1,x2,y2=rectangle(plate)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    area=(x2-x1)*(y2-y1)
    best=0.
    for index,vehicle in enumerate(vehicles):
        vx1,vy1,vx2,vy2=rectangle(vehicle['bbox'])
        overlap=max(0.,min(x2,vx2)-max(x1,vx1))*max(0.,min(y2,vy2)-max(y1,vy1))/area
        best=max(best,overlap)
        if vx1<=cx<=vx2 and vy1<=cy<=vy2 and overlap>=min_intersection:
            return dict(keep=True,vehicle_index=index,intersection_over_plate=overlap,reason='VEHICLE_SUPPORT')
    return dict(keep=False,vehicle_index=None,intersection_over_plate=best,
                reason='NO_VEHICLE_DETECTED' if not vehicles else 'OUTSIDE_VEHICLE')


def filter_plate_detections(detections,vehicles,min_intersection=.8):
    decisions=[dict(detection_index=i,**vehicle_support(d['bbox'],vehicles,min_intersection))
               for i,d in enumerate(detections)]
    return [r['detection_index'] for r in decisions if r['keep']],decisions
