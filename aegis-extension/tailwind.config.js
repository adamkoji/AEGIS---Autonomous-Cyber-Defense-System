/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./*.html",   // Scans all HTML files in the root (like block.html, popup.html)
    "./*.js",     // Scans all JS files in the root (like block.js, popup.js)
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}