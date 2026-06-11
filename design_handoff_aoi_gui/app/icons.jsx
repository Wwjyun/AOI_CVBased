// ============================================================
// AOI Console — icon set（24px stroke line icons）
// ============================================================

function Icon({ children, size = 18, strokeWidth = 1.7, style }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={strokeWidth}
      strokeLinecap="round" strokeLinejoin="round" style={style} aria-hidden="true"
    >{children}</svg>
  );
}

const IcPlay = (p) => <Icon {...p}><path d="M7 5l12 7-12 7V5z" /></Icon>;
const IcImage = (p) => (
  <Icon {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <circle cx="9" cy="10" r="1.6" />
    <path d="M5 18l5-5 3 3 3-3 3 3" />
  </Icon>
);
const IcRecipe = (p) => (
  <Icon {...p}>
    <path d="M7 3h8l4 4v14H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z" />
    <path d="M15 3v4h4" />
    <path d="M9 12h6M9 16h6" />
  </Icon>
);
const IcDesigner = (p) => (
  <Icon {...p}>
    <path d="M4 20l4-1L19 8l-3-3L5 16l-1 4z" />
    <path d="M14 7l3 3" />
  </Icon>
);
const IcTable = (p) => (
  <Icon {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M3 10h18M9 10v10M15 10v10" />
  </Icon>
);
const IcGear = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="3" />
    <path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1" />
  </Icon>
);
const IcFolder = (p) => (
  <Icon {...p}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
  </Icon>
);
const IcCheck = (p) => <Icon {...p}><path d="M4 12.5l5 5L20 6.5" /></Icon>;
const IcX = (p) => <Icon {...p}><path d="M6 6l12 12M18 6L6 18" /></Icon>;
const IcChevronD = (p) => <Icon {...p}><path d="M6 9l6 6 6-6" /></Icon>;
const IcChevronR = (p) => <Icon {...p}><path d="M9 6l6 6-6 6" /></Icon>;
const IcZoomIn = (p) => (
  <Icon {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.5-4.5M8 11h6M11 8v6" />
  </Icon>
);
const IcZoomOut = (p) => (
  <Icon {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.5-4.5M8 11h6" />
  </Icon>
);
const IcFit = (p) => (
  <Icon {...p}>
    <path d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5" />
  </Icon>
);
const IcSave = (p) => (
  <Icon {...p}>
    <path d="M5 3h11l5 5v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z" />
    <path d="M8 3v5h7V3M7 21v-7h10v7" />
  </Icon>
);
const IcEye = (p) => (
  <Icon {...p}>
    <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z" />
    <circle cx="12" cy="12" r="2.8" />
  </Icon>
);
const IcCrosshair = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="7" />
    <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
  </Icon>
);
const IcUpload = (p) => (
  <Icon {...p}>
    <path d="M12 16V4M7 9l5-5 5 5" />
    <path d="M4 20h16" />
  </Icon>
);
const IcHistory = (p) => (
  <Icon {...p}>
    <path d="M3.5 12a8.5 8.5 0 1 0 2.5-6L3.5 8.5" />
    <path d="M3.5 4v4.5H8" />
    <path d="M12 8v4.5l3 2" />
  </Icon>
);
const IcLayers = (p) => (
  <Icon {...p}>
    <path d="M12 3l9 5-9 5-9-5 9-5z" />
    <path d="M3 13l9 5 9-5" />
  </Icon>
);

Object.assign(window, {
  Icon, IcPlay, IcImage, IcRecipe, IcDesigner, IcTable, IcGear, IcFolder,
  IcCheck, IcX, IcChevronD, IcChevronR, IcZoomIn, IcZoomOut, IcFit, IcSave,
  IcEye, IcCrosshair, IcUpload, IcHistory, IcLayers,
});
