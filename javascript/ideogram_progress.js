// Reuse Forge's overall bar; add exactly one current-image bar for Ideogram.
onUiLoaded(function () {
    const original = window.requestProgress;
    if (!original || original._i4Dual) return;
    function wrapped(id, container, gallery, atEnd, onProgress, ...rest) {
        let current = null;
        function cleanup() {
            if (current) current.remove();
            current = null;
            if (gallery) gallery.classList.remove('pi-i4-developing');
        }
        function progress(res) {
            const match = /^Ideogram 4 \| Image (\d+)\/(\d+) \| Step (\d+)\/(\d+)$/.exec(res.textinfo || '');
            if (match && res.active) {
                if (gallery) gallery.classList.add('pi-i4-developing');
                if (!current) {
                    current = document.createElement('div');
                    current.className = 'progressDiv pi-i4-current';
                    current.setAttribute('role', 'progressbar');
                    current.setAttribute('aria-label', 'Current generation');
                    current.setAttribute('aria-valuemin', '0');
                    current.setAttribute('aria-valuemax', '100');
                    current.appendChild(document.createElement('div')).className = 'progress';
                    container.parentNode.insertBefore(current, container);
                }
                current.style.display = opts.show_progressbar ? 'block' : 'none';
                const pct = Math.min(100, 100 * Number(match[3]) / Math.max(1, Number(match[4])));
                current.setAttribute('aria-valuenow', String(pct));
                current.firstChild.style.width = pct + '%';
                current.firstChild.style.transition = 'width 0.35s ease';
                current.firstChild.textContent = `Current image ${match[1]}/${match[2]}: ${Math.round(pct)}%`;
                const overall = [...container.parentNode.children].find(el => el !== current && el.classList.contains('progressDiv'));
                if (overall && overall.firstChild) {
                    overall.firstChild.textContent = `Overall: ${Math.round((res.progress || 0) * 100)}%`;
                }
            } else {
                cleanup();
            }
            if (onProgress) onProgress(res);
        }
        return original(id, container, gallery, (...args) => {
            cleanup();
            if (atEnd) atEnd(...args);
        }, progress, ...rest);
    }
    wrapped._i4Dual = true;
    window.requestProgress = wrapped;
});
