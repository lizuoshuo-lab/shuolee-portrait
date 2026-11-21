from pathlib import Path
H = Path(__file__).parent
source = (H/'review_native_photometric123.py').read_text(encoding='utf-8')
assert source.count("O=D/'native-photometric-contrast'") == 1
source = source.replace("O=D/'native-photometric-contrast'", "O=D/'native-affine-contrast'")
source = source.replace("'photometricB'", "'affineB'").replace("'photometricC'", "'affineC'")
exec(compile(source, str(__file__), 'exec'))
