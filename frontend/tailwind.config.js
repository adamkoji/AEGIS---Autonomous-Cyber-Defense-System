/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        lunar: {
          blue: "#8DB9F0",
          blueDeep: "#5B96E5",
          lavender: "#E9E8F8",
          highlight: "#F5F4FC",
          peach: "#F5B69E",
          coral: "#E88D6D",
          navy: "#1E2238",
          navyHover: "#2A304D",
          slate: "#5A6282",
          nightStart: "#0D111E",
          nightEnd: "#151B30",
          nightCard: "#171C30",
        },
      },
      fontFamily: {
        serif: ["'Cormorant Garamond'", "Georgia", "serif"],
        sans: ["'Plus Jakarta Sans'", "'Inter'", "system-ui", "sans-serif"],
      },
      borderRadius: {
        "3xl": "28px",
        "4xl": "32px",
      },
    },
  },
  plugins: [],
};
