const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function element() {
    return {style: {}, children: [], attributes: {}, className: '',
        classList: {contains(name) { return name === 'progressDiv'; }},
        setAttribute(k, v) { this.attributes[k] = v; },
        appendChild(child) { this.children.push(child); this.firstChild = child; return child; },
        insertBefore(child) { this.children.push(child); child.parentNode = this; },
        remove() { this.parentNode.children = this.parentNode.children.filter(x => x !== this); }};
}
let report, finish;
const context = {document: {createElement: element}, opts: {show_progressbar: true},
    onUiLoaded(fn) { fn(); }, window: {requestProgress(id, c, g, end, progress) {report = progress; finish = end;}}};
vm.runInNewContext(fs.readFileSync('javascript/ideogram_progress.js', 'utf8'), context);
const parent = element(), stock = element();
stock.appendChild(element()); parent.children.push(stock);
let ended = false;
context.window.requestProgress('task', {parentNode: parent}, null, () => ended = true);
report({active: true, textinfo: 'Ideogram 4 | Image 1/2 | Step 6/12', progress: .25});
assert.equal(parent.children.length, 2);
assert.equal(parent.children[1].firstChild.style.width, '50%');
assert.equal(stock.firstChild.textContent, 'Overall: 25%');
report({active: true, textinfo: 'Ideogram 4 | Image 2/2 | Step 0/12', progress: .5});
assert.equal(parent.children.length, 2);
assert.equal(parent.children[1].firstChild.style.width, '0%');
assert.equal(stock.firstChild.textContent, 'Overall: 50%');
finish(); assert.equal(parent.children.length, 1); assert.equal(ended, true);
report({active: true, textinfo: 'Other model', progress: .4});
assert.equal(parent.children.length, 1);
console.log('Dual progress: passed');
