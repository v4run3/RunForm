/**
 * Device-aware session persistence manager.
 *
 * Rules:
 * - Mobile devices: Once signed in, stay signed in across browser launches
 *   until the user explicitly logs out (stored in localStorage).
 * - Laptop / Desktop: Every time the user starts the app in a new session/window,
 *   they are shown the Landing Page with Login & Signup available
 *   (stored in sessionStorage so it clears when browser/session restarts).
 */

const MOBILE_SESSION_KEY = 'runform_mobile_session';
const LAPTOP_SESSION_KEY = 'runform_laptop_session';
const LAPTOP_REMEMBER_KEY = 'runform_laptop_persistent';

/**
 * Detect whether the client is a mobile device (phone / tablet).
 */
export function isMobileDevice() {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return false;
  }
  const userAgent = navigator.userAgent || navigator.vendor || window.opera || '';
  const isMobileUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|mobile|CriOS/i.test(userAgent);
  const isTouch = 'ontouchstart' in window || (navigator.maxTouchPoints && navigator.maxTouchPoints > 0);
  const isSmallScreen = window.innerWidth <= 768;

  return isMobileUA || (isTouch && isSmallScreen);
}

/**
 * Retrieve the current session based on device-specific persistence rules.
 */
export function getStoredSession() {
  if (typeof window === 'undefined') return null;

  if (isMobileDevice()) {
    // Mobile: persistent storage across app launches
    const mobileData = localStorage.getItem(MOBILE_SESSION_KEY);
    if (mobileData) {
      try {
        return JSON.parse(mobileData);
      } catch {
        localStorage.removeItem(MOBILE_SESSION_KEY);
      }
    }
    return null;
  }

  // Laptop / Desktop:
  // By default, every time the laptop browser is started fresh, sessionStorage is empty.
  // This guarantees the landing page is shown on app launch.
  const sessionData = sessionStorage.getItem(LAPTOP_SESSION_KEY);
  if (sessionData) {
    try {
      return JSON.parse(sessionData);
    } catch {
      sessionStorage.removeItem(LAPTOP_SESSION_KEY);
    }
  }

  // Check if laptop user explicitly requested persistent "Remember me"
  const laptopPersistent = localStorage.getItem(LAPTOP_REMEMBER_KEY);
  if (laptopPersistent) {
    try {
      const parsed = JSON.parse(laptopPersistent);
      // Copy to sessionStorage for the active session
      sessionStorage.setItem(LAPTOP_SESSION_KEY, laptopPersistent);
      return parsed;
    } catch {
      localStorage.removeItem(LAPTOP_REMEMBER_KEY);
    }
  }

  return null;
}

/**
 * Save user authentication session according to device policy.
 *
 * @param {object} session - { token, user: { id, name, email } }
 * @param {boolean} rememberOnLaptop - Optional explicit remember flag on laptop
 */
export function saveSession(session, rememberOnLaptop = false) {
  if (typeof window === 'undefined' || !session) return;
  const serialized = JSON.stringify(session);

  if (isMobileDevice()) {
    // Mobile: automatically persistent in localStorage
    localStorage.setItem(MOBILE_SESSION_KEY, serialized);
  } else {
    // Laptop / Desktop: store in sessionStorage so restarting browser returns to landing page
    sessionStorage.setItem(LAPTOP_SESSION_KEY, serialized);
    if (rememberOnLaptop) {
      localStorage.setItem(LAPTOP_REMEMBER_KEY, serialized);
    } else {
      localStorage.removeItem(LAPTOP_REMEMBER_KEY);
    }
    // Clean up mobile key if user previously used responsive mode testing
    localStorage.removeItem(MOBILE_SESSION_KEY);
  }
}

/**
 * Clear all authentication sessions across all storage keys.
 */
export function clearSession() {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(MOBILE_SESSION_KEY);
  localStorage.removeItem(LAPTOP_REMEMBER_KEY);
  localStorage.removeItem('runform_session'); // Legacy cleanup
  sessionStorage.removeItem(LAPTOP_SESSION_KEY);
}
