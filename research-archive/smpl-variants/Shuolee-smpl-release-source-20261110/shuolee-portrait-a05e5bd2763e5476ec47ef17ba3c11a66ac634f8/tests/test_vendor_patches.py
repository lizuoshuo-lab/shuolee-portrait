"""Headless import controls; these checks do not execute GPU model inference."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class VendorPatchTests(unittest.TestCase):
    def test_gvhmr_body_module_imports_without_turtle_or_tkinter(self):
        # Lightweight API stubs isolate the real module's import statements;
        # they do not claim that real Torch/SMPL-X inference has been verified.
        torch = types.ModuleType('torch')
        nn = types.ModuleType('torch.nn')
        nn.Module = object
        torch.nn = nn
        smplx = types.ModuleType('smplx')
        smplx.SMPL = smplx.SMPLH = smplx.SMPLX = object
        vertex_ids = types.ModuleType('smplx.vertex_ids')
        vertex_ids.vertex_ids = {}
        utils = types.ModuleType('smplx.utils')
        utils.Struct = object
        actual_import = builtins.__import__
        def headless_import(name, *args, **kwargs):
            if name.split('.')[0] in ['turtle', 'tkinter', '_tkinter']:
                raise ModuleNotFoundError(f'Headless control rejects {name}')
            return actual_import(name, *args, **kwargs)
        modules = {'torch': torch, 'torch.nn': nn, 'smplx': smplx,
                   'smplx.vertex_ids': vertex_ids, 'smplx.utils': utils}
        with patch.dict(sys.modules, modules), patch('builtins.__import__', headless_import):
            namespace = runpy.run_path(str(ROOT / 'third_party/gvhmr/hmr4d/utils/body_model/body_model.py'))
        self.assertIn('BodyModel', namespace)

    def test_numpy_SH_helper_imports_without_torch(self):
        actual_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name.split('.')[0] == 'torch':
                raise ModuleNotFoundError('Torch intentionally absent in this CPU control')
            return actual_import(name, *args, **kwargs)
        with patch('builtins.__import__', no_torch):
            namespace = runpy.run_path(str(ROOT / 'src/avatar_core/utils/sh_utils.py'))
        rgb = np.array([[0., .5, 1.], [.25, .75, .125]])
        np.testing.assert_allclose(namespace['SH2RGB'](namespace['RGB2SH'](rgb)), rgb)

    def test_patch_receipt_keeps_all_function_and_class_AST_unchanged(self):
        receipt = json.loads((ROOT / 'provenance/vendor-patches.json').read_text(encoding='utf-8'))
        for entry in receipt['patches']:
            tree = ast.parse((ROOT / entry['path']).read_text(encoding='utf-8'))
            definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            digest = hashlib.sha256(ast.dump(ast.Module(body=definitions, type_ignores=[]), include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest, entry['definition_ast_sha256'], entry['path'])
            self.assertTrue(entry['definitions_unchanged'])
            self.assertFalse(entry['license_notice_modified'])


if __name__ == '__main__':
    unittest.main()
