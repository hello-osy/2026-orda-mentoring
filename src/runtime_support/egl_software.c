/* Optional process-local workaround for Mesa ARM VMs exposing an unusable DRM
 * device before llvmpipe. Ogre 2 enumerates that DRM device even in software
 * mode. Expose only EGL_MESA_device_software; all other EGL calls are untouched.
 * Compile: cc -shared -fPIC -O2 egl_software.c -o libegl_software.so -ldl -pthread
 * Only preload into Gazebo when environment.egl_software_only is true.
 * Requires EGL headers (libegl-dev), libc headers and a C compiler.
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#include <string.h>
#include <pthread.h>

static pthread_once_t load_once = PTHREAD_ONCE_INIT;
static PFNEGLGETPROCADDRESSPROC original_get_proc;

static void load_egl(void) {
    /* Keep one reference for this process; function pointers stay valid. */
    void *library = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library) original_get_proc = dlsym(library, "eglGetProcAddress");
}

static PFNEGLGETPROCADDRESSPROC get_original(void) {
    pthread_once(&load_once, load_egl);
    return original_get_proc;
}

static EGLBoolean software_devices(EGLint max_devices, EGLDeviceEXT *devices,
                                    EGLint *num_devices) {
    PFNEGLGETPROCADDRESSPROC get_proc = get_original();
    if (!get_proc) return EGL_FALSE;
    PFNEGLQUERYDEVICESEXTPROC query = (PFNEGLQUERYDEVICESEXTPROC)get_proc("eglQueryDevicesEXT");
    PFNEGLQUERYDEVICESTRINGEXTPROC string = (PFNEGLQUERYDEVICESTRINGEXTPROC)get_proc("eglQueryDeviceStringEXT");
    EGLDeviceEXT all[64];
    EGLint count = 0, found = 0;
    if (!query || !string || !query(64, all, &count)) return EGL_FALSE;
    for (EGLint i = 0; i < count; i++) {
        const char *extensions = string(all[i], EGL_EXTENSIONS);
        if (extensions && strstr(extensions, "EGL_MESA_device_software")) {
            if (devices && found < max_devices) devices[found] = all[i];
            found++;
        }
    }
    if (num_devices) *num_devices = devices && found > max_devices ? max_devices : found;
    return EGL_TRUE;
}

__eglMustCastToProperFunctionPointerType eglGetProcAddress(const char *name) {
    PFNEGLGETPROCADDRESSPROC real = get_original();
    if (!strcmp(name, "eglQueryDevicesEXT"))
        return (__eglMustCastToProperFunctionPointerType)software_devices;
    return real ? real(name) : NULL;
}

EGLBoolean eglQueryDevicesEXT(EGLint max_devices, EGLDeviceEXT *devices, EGLint *num_devices) {
    return software_devices(max_devices, devices, num_devices);
}
