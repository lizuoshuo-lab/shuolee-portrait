"""Install a completed, user-downloaded official SMPL archive locally.

Only the neutral template and license/readme documents are extracted. Neither
the template nor the archive belongs in the website's public asset directory.
This step inventories pickle globals without executing the pickle.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickletools
import zipfile

LAB = Path(__file__).resolve().parents[2]
OUT = LAB / 'outputs/dynamic-human-smpl-010'
PRIVATE = LAB / 'data/licensed/smpl'
WINDOWS_MODEL = LAB / 'repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl'
LINUX_MODEL = Path(r'\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\dynamic-human\repos\drivestudio\smpl_models\SMPL_NEUTRAL.pkl')


def sha(path):
    with Path(path).open('rb') as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def put_identical(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'Refusing to replace different existing model: {path}')
    else:
        path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    assert archive.suffix.lower() == '.zip', 'Wait for the browser to finish the ZIP download'
    size_before = archive.stat().st_size
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        candidates = [m for m in members if Path(m.filename).name.lower() in (
            'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl', 'smpl_neutral.pkl')]
        assert len(candidates) == 1, [m.filename for m in candidates]
        corrupt = package.testzip()
        assert corrupt is None, f'ZIP CRC failure: {corrupt}'
        member = candidates[0]
        assert 1_000_000 < member.file_size < 500_000_000, member.file_size
        content = package.read(member)
        # Protocol-0 Python-2 NumPy data uses byte strings beyond ASCII. Only
        # scan opcodes; do not decode their data strings or execute the pickle.
        string_argument = next(op.arg for op in pickletools.opcodes if op.name == 'STRING')
        original_reader = string_argument.reader
        try:
            string_argument.reader = lambda stream: pickletools.read_stringnl(stream, decode=False)
            globals_used = sorted({arg for op, arg, pos in pickletools.genops(content) if op.name == 'GLOBAL'})
        finally:
            string_argument.reader = original_reader
        PRIVATE.mkdir(parents=True, exist_ok=True)
        original = PRIVATE / Path(member.filename).name
        put_identical(original, content)
        docs = []
        for index, entry in enumerate(members):
            name = Path(entry.filename).name
            if not entry.is_dir() and ('license' in name.lower() or 'readme' in name.lower()):
                target = PRIVATE / 'archive-documents' / f'{index:03d}-{name}'
                put_identical(target, package.read(entry))
                docs.append(dict(member=entry.filename, local_path=str(target), sha256=sha(target)))
    assert archive.stat().st_size == size_before, 'Archive changed while being checked'
    put_identical(WINDOWS_MODEL, content)
    put_identical(LINUX_MODEL, content)
    digest = hashlib.sha256(content).hexdigest()
    assert sha(WINDOWS_MODEL) == sha(LINUX_MODEL) == digest
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(status='archive_crc_and_identical_local_copies_passed_runtime_validation_pending',
        official_download='https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip',
        archive=str(archive), archive_bytes=size_before, archive_sha256=sha(archive),
        member=member.filename, model_bytes=len(content), model_sha256=digest,
        original=str(original), windows_model=str(WINDOWS_MODEL), linux_model=str(LINUX_MODEL),
        pickle_globals=globals_used, license_documents=docs,
        validation_boundary='ZIP integrity and byte-identical copies only; no model forward pass or mesh alignment claim.',
        publication='Model and license-restricted archive remain local; never sync these assets to the website.')
    (OUT / 'template-installation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
