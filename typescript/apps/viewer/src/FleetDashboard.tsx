// FleetDashboard.tsx — Real-time fleet status dashboard
//
// Displays device status, active jobs, and chunk completion.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect } from 'react';
import type { DeviceCapability, WorkChunk } from '@bonhomme/flexaidds';

interface FleetState {
  devices: DeviceCapability[];
  activeChunks: WorkChunk[];
  completedChunks: number;
  totalChunks: number;
}

export function FleetDashboard() {
  const [fleet, setFleet] = useState<FleetState>({
    devices: [],
    activeChunks: [],
    completedChunks: 0,
    totalChunks: 0,
  });

  // Poll fleet status from iCloud watcher or WebSocket
  useEffect(() => {
    // Placeholder: in production, this connects to the fleet coordinator
    // via CloudKit JS or a WebSocket relay from the Mac home base
  }, []);

  const thermalColor = (state: string) => {
    switch (state) {
      case 'nominal': return '#4ade80';
      case 'fair': return '#fbbf24';
      case 'serious': return '#f97316';
      case 'critical': return '#ef4444';
      default: return '#9ca3af';
    }
  };

  return (
    <div>
      <h2>Bonhomme Fleet</h2>

      <section>
        <h3>Devices ({fleet.devices.length})</h3>
        {fleet.devices.length === 0 ? (
          <p style={{ color: '#666' }}>
            No fleet devices connected. Enable iCloud Drive and fleet mode on each device.
          </p>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '1rem' }}>
            {fleet.devices.map((device) => (
              <div key={device.deviceID} style={{
                border: '1px solid #333',
                borderRadius: '8px',
                padding: '1rem',
                background: '#1a1a2e',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <strong>{device.model}</strong>
                  <span style={{
                    width: '12px', height: '12px', borderRadius: '50%',
                    backgroundColor: thermalColor(device.thermalState),
                    display: 'inline-block',
                  }} />
                </div>
                <div style={{ fontSize: '0.85rem', color: '#999', marginTop: '0.5rem' }}>
                  <div>{device.estimatedTFLOPS.toFixed(1)} TFLOPS</div>
                  <div>{device.availableMemoryGB.toFixed(0)} GB RAM</div>
                  <div>Weight: {(device.computeWeight * 100).toFixed(0)}%</div>
                  <div>Thermal: {device.thermalState}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {fleet.totalChunks > 0 && (
        <section>
          <h3>Job Progress</h3>
          <div style={{ background: '#333', borderRadius: '4px', height: '24px', overflow: 'hidden' }}>
            <div style={{
              background: '#4ade80',
              height: '100%',
              width: `${(fleet.completedChunks / fleet.totalChunks) * 100}%`,
              transition: 'width 0.3s',
            }} />
          </div>
          <p>{fleet.completedChunks} / {fleet.totalChunks} chunks completed</p>
        </section>
      )}

      {fleet.activeChunks.length > 0 && (
        <section>
          <h3>Active Chunks</h3>
          <table>
            <thead>
              <tr>
                <th>Chunk</th>
                <th>Status</th>
                <th>Device</th>
                <th>Chromosomes</th>
              </tr>
            </thead>
            <tbody>
              {fleet.activeChunks.map((chunk) => (
                <tr key={chunk.id}>
                  <td>{chunk.index + 1}/{chunk.totalChunks}</td>
                  <td>{chunk.status}</td>
                  <td>{chunk.claimedBy ?? '—'}</td>
                  <td>{chunk.gaParameters.numChromosomes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
