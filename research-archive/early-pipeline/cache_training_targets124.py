"""Two-line input interface adaptation for new experiments only."""
import ast


def adapt(source):
    marker='    class Images:\n'
    old='cams[i].original_image = pixels[i].float().div(255) * masks[i].float().div(255)'
    assert source.count(marker)==1 and source.count(old)==1
    result=source.replace(marker,
        "    from native_target_cache124 import NativeTargetCache\n"
        "    target_cache = NativeTargetCache(pixels, masks, [r['frame'] for r in rows])\n"+marker)
    result=result.replace(old,"cams[i].original_image = target_cache.get(i, rows[i]['frame'])")
    ast.parse(result)
    return result
