/* Parity driver: decode base32 payloads with the SHIPPED browser decoder.
 *
 * Invoked by test_parity.py -- see there for why this exists. Reads
 *   argv[2] = path to the .bin model table
 *   argv[3] = path to a JSON file: [{name, b32}, ...]
 * and writes JSON to stdout: [{name, text}] or [{name, error}].
 */
'use strict';
const fs = require('fs');
const path = require('path');

const decoderPath = process.env.ARTCODE_JS ||
  path.join(__dirname, '..', '..', '..', 'jakobrees.com', 'public', 'a', 'artcode.js');
const Artcode = require(decoderPath);

const tableBuf = fs.readFileSync(process.argv[2]);
const model = Artcode.parseTable(
  tableBuf.buffer.slice(tableBuf.byteOffset, tableBuf.byteOffset + tableBuf.byteLength));

const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = cases.map(c => {
  try {
    return { name: c.name, text: Artcode.decodeFragment(c.b32, model) };
  } catch (e) {
    return { name: c.name, error: String(e && e.message || e) };
  }
});
process.stdout.write(JSON.stringify(out));
