function showSource(type) {
  ['usb', 'ip', 'file'].forEach(t => {
    const el = document.getElementById('src-' + t);
    if (el) el.classList.toggle('hidden', t !== type);
  });
  const rtspInput = document.querySelector('input[name="rtsp_url"]');
  const fileInput = document.querySelector('input[name="video_file"]');
  if (rtspInput) rtspInput.required = (type === 'ip');
  if (fileInput) fileInput.required = (type === 'file');
}
