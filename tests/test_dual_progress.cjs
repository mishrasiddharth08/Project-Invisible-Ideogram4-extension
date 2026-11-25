const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const extensionProgress = fs.readFileSync(
    path.resolve(__dirname, '..', 'javascript', 'ideogram_progress.js'), 'utf8');
const sentinel = function requestProgress() {};
const extensionContext = {window: {requestProgress: sentinel}};
vm.runInNewContext(extensionProgress, extensionContext);
assert.equal(extensionContext.window.requestProgress, sentinel,
    'extension must not wrap Forge requestProgress');

const stockProgress = fs.readFileSync(
    path.resolve(__dirname, '..', '..', '..', 'javascript', 'progressbar.js'), 'utf8');

function element() {
    return {
        children: [], className: '', style: {}, textContent: '',
        classList: {contains() { return false; }},
        appendChild(child) { child.parentNode = this; this.children.push(child); return child; },
        insertBefore(child, before) {
            child.parentNode = this;
            const index = this.children.indexOf(before);
            this.children.splice(index < 0 ? this.children.length : index, 0, child);
        },
        removeChild(child) {
            this.children.splice(this.children.indexOf(child), 1);
            child.parentNode = null;
        },
    };
}

function runStockBar(showProgressbar) {
    const responses = [
        {active: true, progress: 0.42, eta: 65, textinfo: null},
        {active: false, progress: 0.42, eta: 65, textinfo: null},
    ];
    const timers = [];
    let ended = 0;

    class XMLHttpRequest {
        open() {}
        setRequestHeader() {}
        send() {
            this.readyState = 4;
            this.status = 200;
            this.responseText = JSON.stringify(responses.shift());
            this.onreadystatechange();
        }
    }

    const context = {
        console, XMLHttpRequest,
        document: {title: 'Forge', createElement: element},
        navigator: {},
        opts: {
            show_progressbar: showProgressbar,
            show_progress_in_title: false,
            prevent_screen_sleep_during_generation: false,
            live_preview_refresh_period: 500,
        },
        onUiLoaded(fn) { fn(); },
        setTimeout(fn) { timers.push(fn); },
    };
    context.window = context;
    vm.runInNewContext(stockProgress, context);

    const parent = element();
    const container = element();
    parent.appendChild(container);
    context.window.requestProgress('task', container, null, () => ended++);

    const bars = parent.children.filter(child => child.className === 'progressDiv');
    assert.equal(bars.length, 1, 'Forge must render exactly one stock progress bar');
    assert.equal(bars[0].style.display, showProgressbar ? 'block' : 'none');
    assert.equal(bars[0].children[0].style.width, '42%');
    assert.equal(bars[0].children[0].textContent, '42% ETA: 01:05');

    timers.shift()();
    assert.equal(parent.children.filter(child => child.className === 'progressDiv').length, 0);
    assert.equal(ended, 1, 'stock cleanup must call atEnd once');
}

runStockBar(true);
runStockBar(false);
console.log('Native stock progress: passed');
