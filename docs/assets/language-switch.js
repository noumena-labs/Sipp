(function () {
    const ZH = "zh";

    function currentLanguage() {
        const lang = document.documentElement.lang.toLowerCase();
        return lang.startsWith(ZH) ? ZH : "en";
    }

    function bookRootPath() {
        return new URL(path_to_root || "./", window.location.href).pathname;
    }

    function targetHref() {
        const lang = currentLanguage();
        const rootPath = bookRootPath();
        const relativePath = window.location.pathname.slice(rootPath.length);
        const suffix = window.location.search + window.location.hash;

        if (lang === ZH) {
            return rootPath.replace(/\/zh\/$/, "/") + relativePath + suffix;
        }

        return rootPath + "zh/" + relativePath + suffix;
    }

    function mountLanguageSwitch() {
        if (document.querySelector(".sipp-language-switch")) return;

        const lang = currentLanguage();
        const target = targetHref();

        const link = document.createElement("a");
        link.className = "sipp-language-switch";
        link.href = target;
        link.title = lang === ZH ? "Switch to English" : "切换到中文";
        link.setAttribute("aria-label", link.title);

        // The inline fill:none wins over the stock `.fa-svg svg` fill rule,
        // which would otherwise fill these stroke-drawn glyphs solid.
        const enSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" style="fill:none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 6H5v12h5"/><line x1="5" y1="12" x2="9" y2="12"/><path d="M14 18V6L19 18V6"/></svg>`;

        const zhSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" style="fill:none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="8" width="14" height="8" rx="1"/><line x1="12" y1="3" x2="12" y2="21"/></svg>`;

        // If we are currently on ZH, we show the EN icon to switch to English.
        // If we are currently on EN, we show the ZH icon to switch to Chinese.
        // The anchor + fa-svg span mirrors the print/git links in
        // .right-buttons, so the stock menu-bar rules handle sizing, spacing,
        // and color.
        link.innerHTML = `<span class="fa-svg">${lang === ZH ? enSvg : zhSvg}</span>`;

        const rightButtons = document.querySelector(".right-buttons");
        if (rightButtons) {
            rightButtons.prepend(link);
        }

        // Warm the counterpart page so the switch navigation is instant.
        const prefetch = document.createElement("link");
        prefetch.rel = "prefetch";
        prefetch.href = target;
        document.head.appendChild(prefetch);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountLanguageSwitch);
    } else {
        mountLanguageSwitch();
    }
})();
