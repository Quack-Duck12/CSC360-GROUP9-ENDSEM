package com.classcounter.util;

import org.opencv.core.Core;

/** Loads the OpenCV native library once (openpnp bundle if present, else system library). */
public final class NativeLoader {
    private static boolean loaded;
    private NativeLoader() {}

    /**
     * Loads the OpenCV native library once. Tries the openpnp bundle first (works on Windows, macOS and Linux
     * without a system install) and falls back to a system-installed library.
     */
    public static synchronized void load() {
        if (loaded) return;
        try {
            Class.forName("nu.pattern.OpenCV").getMethod("loadLocally").invoke(null);
        } catch (Throwable bundled) {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        }
        loaded = true;
    }
}
