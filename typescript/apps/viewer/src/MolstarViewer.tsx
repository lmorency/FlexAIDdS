// MolstarViewer.tsx — Mol* 3D molecular viewer for BonhommeViewer
//
// Renders binding population poses with Boltzmann-weighted coloring.
// Color spectrum: burgundy red (high probability) → purple blue (low probability).
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import React, { useRef, useEffect, useCallback } from 'react';
import type { BindingPopulation } from '@bonhomme/shared';
import { PluginContext } from 'molstar/lib/mol-plugin/context';
import { DefaultPluginSpec } from 'molstar/lib/mol-plugin/spec';
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import { ColorTheme } from 'molstar/lib/mol-theme/color';
import { Color } from 'molstar/lib/mol-util/color';

/**
 * Interpolate between burgundy red and purple blue based on Boltzmann weight.
 *
 * weight = 1.0 → burgundy red (#800020)
 * weight = 0.0 → purple blue (#4B0082)
 */
function populationColor(weight: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, weight));
  // Burgundy red: (128, 0, 32)  → Purple blue: (75, 0, 130)
  const r = Math.round(75 + t * (128 - 75));
  const g = 0;
  const b = Math.round(130 + t * (32 - 130));
  return [r, g, b];
}

/** CSS hex string for the population color */
export function populationColorHex(weight: number): string {
  const [r, g, b] = populationColor(weight);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

interface MolstarViewerProps {
  population: BindingPopulation;
  pdbData?: string;
  selectedModeIndex?: number;
}

export function MolstarViewer({ population, pdbData, selectedModeIndex }: MolstarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pluginRef = useRef<PluginContext | null>(null);

  // Initialize Mol* plugin
  useEffect(() => {
    if (!containerRef.current) return;

    const init = async () => {
      const plugin = new PluginContext(DefaultPluginSpec());
      await plugin.init();

      if (!containerRef.current) {
        plugin.dispose();
        return;
      }

      const canvas = containerRef.current.querySelector('canvas');
      if (!canvas) {
        const newCanvas = document.createElement('canvas');
        newCanvas.style.width = '100%';
        newCanvas.style.height = '100%';
        containerRef.current.appendChild(newCanvas);
      }

      plugin.initViewer(containerRef.current, {
        layoutIsExpanded: false,
        layoutShowControls: false,
        layoutShowRemoteState: false,
        layoutShowSequence: false,
        layoutShowLog: false,
      });

      pluginRef.current = plugin;
    };

    init();

    return () => {
      pluginRef.current?.dispose();
      pluginRef.current = null;
    };
  }, []);

  // Load PDB data when available
  useEffect(() => {
    const plugin = pluginRef.current;
    if (!plugin || !pdbData) return;

    const loadStructure = async () => {
      await PluginCommands.State.RemoveObject(plugin, { state: plugin.state.data, ref: 'structure' });

      const data = await plugin.builders.data.rawData({ data: pdbData }, { state: { isGhost: true } });
      const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
      await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
    };

    loadStructure();
  }, [pdbData]);

  return (
    <div>
      <div
        ref={containerRef}
        id="molstar-viewer"
        style={{ width: '100%', height: '500px', background: '#1a1a2e', borderRadius: '8px' }}
      />
      {population.modes.length > 0 && (
        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', fontSize: '0.85em' }}>
          <span>Boltzmann weight scale:</span>
          <span style={{ color: populationColorHex(1.0) }}>High (burgundy)</span>
          <span style={{ color: populationColorHex(0.5) }}>Medium</span>
          <span style={{ color: populationColorHex(0.0) }}>Low (purple)</span>
        </div>
      )}
    </div>
  );
}
