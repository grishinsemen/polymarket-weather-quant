"use client";

import React, { useCallback, useEffect, useState } from "react";
import { StationRadarTerminal, type WsStatus } from "@/components/StationRadarTerminal";
import type { RadarStateResponse } from "@/types/radar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EMPTY_STATE: RadarStateResponse = {
  status: "ok",
  active_model_version: "ensemble-3layer-v1",
  active_strategy_version: "conservative-kelly-100-v1",
  server_time_utc: new Date().toISOString(),
  last_sync_utc: null,
  last_sync_error: null,
  eligible_count: 0,
  quarantined_count: 0,
  metar_station_count: 0,
  stations: [],
  quarantined_events: [],
  catalog_stations: [],
};

export function LiveRadarShell() {
  const [state, setState] = useState<RadarStateResponse>(EMPTY_STATE);
  const [wsStatus, setWsStatus] = useState<WsStatus>("connecting");
  const [isSyncing, setIsSyncing] = useState(false);

  const fetchInitialState = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/state`, { cache: "no-store" });
      if (resp.ok) {
        const data = (await resp.json()) as RadarStateResponse;
        setState(data);
      }
    } catch {
      // Backend may still be booting; WebSocket reconnects below
    }
  }, []);

  const handleSync = useCallback(async () => {
    setIsSyncing(true);
    try {
      const resp = await fetch(`${API_BASE}/api/stations/sync`, { method: "POST" });
      if (resp.ok) {
        const data = (await resp.json()) as RadarStateResponse;
        setState(data);
      }
    } finally {
      setIsSyncing(false);
    }
  }, []);

  useEffect(() => {
    void fetchInitialState();

    const wsUrl = API_BASE.replace(/^http/, "ws") + "/ws/stream";
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let isDisposed = false;

    const connect = () => {
      if (isDisposed) return;
      setWsStatus("connecting");
      socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        setWsStatus("connected");
      };
      socket.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data as string) as {
            type?: string;
            payload?: RadarStateResponse;
          };
          if (msg.type === "state_snapshot" && msg.payload) {
            setState(msg.payload);
          }
        } catch {
          // Ignore malformed WS frame
        }
      };
      socket.onclose = () => {
        setWsStatus("disconnected");
        if (!isDisposed) {
          reconnectTimer = setTimeout(connect, 3000);
        }
      };
      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      isDisposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [fetchInitialState]);

  return (
    <StationRadarTerminal
      initialState={state}
      wsStatus={wsStatus}
      onSync={handleSync}
      isSyncing={isSyncing}
    />
  );
}
