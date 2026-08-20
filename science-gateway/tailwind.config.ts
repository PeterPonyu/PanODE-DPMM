import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        brand: '#3d4f7c',
        paper: '#efe6d6',
      },
    },
  },
  plugins: [],
};

export default config;
