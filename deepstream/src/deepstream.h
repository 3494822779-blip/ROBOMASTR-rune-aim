#ifndef __DEEPSTREAM_H__
#define __DEEPSTREAM_H__

#include <nvdsgstutils.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#include "modules/interrupt.h"
#include "modules/perf.h"

static gchar *SOURCE = NULL;
static gchar *INFER_CONFIG = NULL;
static guint STREAMMUX_BATCH_SIZE = 1;
static guint STREAMMUX_WIDTH = 1920;
static guint STREAMMUX_HEIGHT = 1080;
static guint GPU_ID = 0;

static guint PERF_MEASUREMENT_INTERVAL_SEC = 5;
static gboolean JETSON = FALSE;

static gint skeleton[][2] = {{1, 2}, {2, 5}, {5, 4}, {4, 1}};

#endif
