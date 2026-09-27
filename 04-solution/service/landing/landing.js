const demoFrame = document.getElementById('demo-frame');
const imageStage = document.getElementById('image-stage');
const frameFallback = document.getElementById('frame-fallback');

if (demoFrame && imageStage && frameFallback) {
  const showFallback = () => {
    demoFrame.hidden = true;
    frameFallback.hidden = false;
    imageStage.classList.add('image-unavailable');
  };
  demoFrame.addEventListener('error', showFallback);
  if (demoFrame.complete && demoFrame.naturalWidth === 0) showFallback();
}
