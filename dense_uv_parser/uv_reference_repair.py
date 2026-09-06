"""Explicit, audited UV correction from a user-selected reference and masks.

This is an opt-in annotation tool, not an automatic parser rule or model update.
The caller defines the beard region and intended support. It never guesses an
object from RGB, changes a checkpoint, or silently applies to other inputs.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


def metadata():
    t=build_simple_uv_topology()
    return ((t.valid&(t.part==0)).numpy(),t.layer.numpy(),t.mirrored_texel.numpy().reshape(-1))


def symmetry_report(uv,scope):
    _,layer,mirror=metadata();pixels=np.asarray(uv).reshape(4096,4).astype(int);s=np.asarray(scope).reshape(-1)
    report={}
    for name,index in [('inner',0),('outer',1)]:
        selected=s&(layer.reshape(-1)==index);ids=np.flatnonzero(selected)
        alpha=pixels[ids,3]!=pixels[mirror[ids],3]
        visible=(pixels[ids,3]>0)|(pixels[mirror[ids],3]>0)
        rgb=np.abs(pixels[ids,:3]-pixels[mirror[ids],:3])
        report[name]={'alpha_mismatched_texels':int(alpha.sum()),'rgb_mismatched_texels':int((rgb.max(1)[visible]>0).sum()),'rgb_max_difference':int(rgb[visible].max()) if visible.any() else 0}
    return report


def evaluate_annotated_review(uv, source):
    """Measure a known development example; never modify predictions or train.

    Identity only selects its explicit review annotation. Symmetry is not a
    universal requirement for other characters or asymmetric accessories.
    """
    root = Path(__file__).resolve().parent.parent
    review_path = root / "dense_uv_parser/regression/v102_uv_symmetry_review.json"
    review = json.loads(review_path.read_text())
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != review["input_sha256"]:
        return None
    beard = np.array(Image.open(root / review["beard_scope"]).convert("L")) > 127
    hair = np.array(Image.open(root / review["hair_scope"]).convert("L")) > 127
    expected = np.array(Image.open(root / review["expected_uv"]).convert("RGBA"))
    metrics = symmetry_report(uv, beard)
    symmetry_passed = all(
        row["alpha_mismatched_texels"] == 0 and row["rgb_mismatched_texels"] == 0
        for row in metrics.values()
    )
    hair_alpha_error = int((uv[hair, 3] != expected[hair, 3]).sum())
    return {
        "annotation": str(review_path.relative_to(root)),
        "beard_symmetry": metrics,
        "beard_symmetry_passed": symmetry_passed,
        "hair_reference_alpha_mismatched_texels": hair_alpha_error,
        "passed": symmetry_passed and hair_alpha_error == 0,
        "scope": "Annotated development example; not an independent generalization score.",
    }


def repair_uv(uv,reference,hair_scope,beard_scope,beard_support,colour_sources):
    head,layer,mirror=metadata()
    arrays=[uv,reference]
    if any(a.shape!=(64,64,4) or a.dtype!=np.uint8 for a in arrays):raise ValueError('Expected uint8 RGBA 64x64 skins')
    masks=[np.asarray(a,dtype=bool) for a in (hair_scope,beard_scope,beard_support,colour_sources)]
    if any(a.shape!=(64,64) for a in masks):raise ValueError('Expected 64x64 masks')
    hair,scope,support,sources=masks
    if ((hair|scope)&~head).any():raise ValueError('This repair is restricted to head UV')
    if (hair&scope).any() or (support&~scope).any():raise ValueError('Overlapping reference/beard regions or support outside region')
    flat_scope=scope.reshape(-1)
    if not np.array_equal(flat_scope,flat_scope[mirror]):raise ValueError('Beard region must include both physical mirror partners')
    result=uv.copy();result[hair]=reference[hair]
    original=uv.reshape(4096,4);pixels=result.reshape(4096,4)
    # Symmetry is explicitly requested by the annotation, not inferred for
    # every character. Union preserves the caller's supported beard cells.
    target=support.reshape(-1)|support.reshape(-1)[mirror]
    outer=(layer.reshape(-1)==1)&flat_scope
    pixels[outer&~target]=0;pixels[outer&target,3]=255
    # Every outer beard cell must have a matching beard substrate in this
    # explicit mixed-layer annotation. Visible beard can remain inner-only.
    target_ids=np.flatnonzero(target)
    for i in target_ids:
        if layer.reshape(-1)[i]==1 and not target[i-32]:raise ValueError('Outer beard support has no corresponding inner beard')
    used=set()
    for i in np.flatnonzero(flat_scope):
        if i in used:continue
        partner=int(mirror[i]);members={int(i),partner}
        if target[i]:
            base=i-32 if layer.reshape(-1)[i]==1 else i
            base_partner=int(mirror[base])
            members={j for j in (base,base_partner,base+32,base_partner+32) if target[j]}
            candidates=[j for j in members if sources.reshape(-1)[j] and original[j,3]>0]
            if not candidates:raise ValueError('No annotated beard material source for mirror pair '+str((i,partner)))
            colour=np.rint(np.median(original[candidates,:3].astype(float),axis=0)).astype(np.uint8)
            for j in members:pixels[j,:3]=colour;pixels[j,3]=255
        elif layer.reshape(-1)[i]==0:
            # The exposed mouth/cheek gap is part of the reviewed outline. Do
            # not fill it with beard; retain a symmetric average of its skin.
            colour=np.rint(original[[i,partner],:3].astype(float).mean(0)).astype(np.uint8)
            pixels[[i,partner],:3]=colour
        used.update(members)
    report={'before':symmetry_report(uv,scope),'after':symmetry_report(result,scope),'hair_reference_exact':bool(np.array_equal(result[hair],reference[hair])),'outside_regions_exact':bool(np.array_equal(result[~(hair|scope)],uv[~(hair|scope)])),'body_exact':bool(np.array_equal(result[16:],uv[16:])),'scope':'Explicit reference-guided repair of an annotated example, not a claim of automatic model generalization.'}
    assert all(v['alpha_mismatched_texels']==0 and v['rgb_mismatched_texels']==0 for v in report['after'].values())
    return result,report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--audit-only',action='store_true');o=p.parse_args();spec=json.loads(o.spec.read_text())
    loaded={}
    for key,entry in spec['assets'].items():
        path=Path(entry['path']);data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Input changed: '+key)
        loaded[key]=np.array(Image.open(path).convert('RGBA' if key in ('uv','reference') else 'L'))
    if o.audit_only:
        report=symmetry_report(loaded['uv'],loaded['beard_scope']>127)
        hair=loaded['hair_scope']>127
        report['hair_reference_alpha_mismatched_texels']=int((loaded['uv'][hair,3]!=loaded['reference'][hair,3]).sum())
        print(json.dumps(report,indent=2));return
    if o.output is None:p.error('--output is required unless --audit-only is used')
    uv,report=repair_uv(loaded['uv'],loaded['reference'],*[loaded[k]>127 for k in ('hair_scope','beard_scope','beard_support','colour_sources')])
    if o.output.exists():raise ValueError('Output already exists; choose a new candidate path')
    o.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(uv).save(o.output)
    report.update(spec=spec,output=str(o.output),output_sha256=hashlib.sha256(o.output.read_bytes()).hexdigest())
    o.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['before']),json.dumps(report['after']),flush=True)

if __name__=='__main__':main()
