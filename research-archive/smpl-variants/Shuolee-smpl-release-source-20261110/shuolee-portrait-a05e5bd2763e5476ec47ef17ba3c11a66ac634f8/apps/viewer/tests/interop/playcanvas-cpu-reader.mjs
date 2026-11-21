import { PlyParser } from 'playcanvas/build/playcanvas/src/framework/parsers/ply.js';

/** Pinned upstream decoder, stopped after load:data and before creating GPU textures. */
export async function decodePlayCanvasCpu(bytes) {
  const stopBeforeGpu = new Error('CPU_PARSE_COMPLETE');
  let decoded;
  await new Promise((resolve, reject) => {
    const parser = new PlyParser({ scene: { gsplatCentersEnabled: false } });
    parser.load({ load: 'unused-model.ply' }, (error) => {
      if (error === stopBeforeGpu && decoded) resolve();
      else reject(error || new Error('The upstream decoder did not emit load:data.'));
    }, {
      data: { reorder: false }, file: { contents: new Response(bytes) },
      fire(event, data) { if (event === 'load:data') { decoded = data; throw stopBeforeGpu; } },
    });
  });
  return decoded;
}
