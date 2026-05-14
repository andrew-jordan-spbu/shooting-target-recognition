function showSource(type) {
  ['usb', 'ip', 'file'].forEach(t => {
    const el = document.getElementById('src-' + t);
    if (el) el.classList.toggle('hidden', t !== type);
  });
}
