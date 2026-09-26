/* CineIQ — Main JS */

// Mobile hamburger menu
document.addEventListener('DOMContentLoaded', () => {
  const btn      = document.getElementById('hamburger-btn');
  const menu     = document.getElementById('mobile-menu');
  const openIcon = document.getElementById('hamburger-icon');
  const closeIcon= document.getElementById('close-icon');

  if (btn && menu) {
    btn.addEventListener('click', () => {
      const isHidden = menu.classList.toggle('hidden');
      btn.setAttribute('aria-expanded', String(!isHidden));
      openIcon.classList.toggle('hidden', !isHidden);
      closeIcon.classList.toggle('hidden', isHidden);
    });
  }
});
