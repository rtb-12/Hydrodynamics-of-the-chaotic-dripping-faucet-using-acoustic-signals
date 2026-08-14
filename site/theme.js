/* Theme selection. Runs synchronously in <head> so the choice is applied
   before first paint, otherwise a stored dark preference flashes light. */
(function () {
  var KEY = 'dfa-theme';
  var root = document.documentElement;

  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* private mode */ }
  if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;

  function label() {
    var t = root.dataset.theme;
    return t === 'light' ? 'Light' : t === 'dark' ? 'Dark' : 'System';
  }

  function apply(next) {
    if (next === 'system') delete root.dataset.theme;
    else root.dataset.theme = next;
    try {
      if (next === 'system') localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, next);
    } catch (e) { /* ignore */ }
    var b = document.getElementById('themeBtn');
    if (b) { b.textContent = label(); b.setAttribute('aria-label', 'Colour theme: ' + label()); }
  }

  document.addEventListener('DOMContentLoaded', function () {
    var b = document.getElementById('themeBtn');
    if (!b) return;
    b.textContent = label();
    b.setAttribute('aria-label', 'Colour theme: ' + label());
    b.addEventListener('click', function () {
      var t = root.dataset.theme;
      apply(t === 'light' ? 'dark' : t === 'dark' ? 'system' : 'light');
    });
  });
})();
