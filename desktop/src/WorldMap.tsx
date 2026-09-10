import { useEffect, useRef, useState } from 'react';
import maplibregl, { type Map as MapInstance, type ExpressionSpecification } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Minus, Plus, Maximize2 } from 'lucide-react';
import { missingColor } from './model';

export type Paint = Record<string, { color: string; label: string }>;
export default function WorldMap({ paint, selected, onSelect, names }: { paint: Paint; selected: string; onSelect: (code: string, name: string) => void; names: Record<string, string> }) {
  const element = useRef<HTMLDivElement>(null);
  const instance = useRef<MapInstance | null>(null);
  const select = useRef(onSelect); select.current = onSelect;
  const values = useRef(paint); values.current = paint;
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const [hover, setHover] = useState<{ x: number; y: number; code: string; name: string } | null>(null);
  const resetWorld = () => instance.current?.fitBounds([[-179, -58], [179, 82]], { padding: { top: 200, bottom: 170, left: 30, right: 30 }, duration: 0 });
  useEffect(() => {
    if (!element.current) return;
    let map: MapInstance;
    try {
      map = new maplibregl.Map({ container: element.current, style: { version: 8, sources: {}, layers: [{ id: 'water', type: 'background', paint: { 'background-color': '#e7ede9' } }] }, center: [12, 34], zoom: 1.45, minZoom: -.5, maxZoom: 6, attributionControl: false, renderWorldCopies: false });
    } catch { setError('The map needs WebGL graphics support. Country search and charts are still available.'); return; }
    instance.current = map;
    map.dragRotate.disable(); map.touchZoomRotate.disableRotation();
    map.addControl(new maplibregl.AttributionControl({ compact: false, customAttribution: 'Made with Natural Earth · country boundaries at overview scale' }));
    map.on('error', e => { setError(`Map could not load: ${e.error.message}`); });
    map.on('load', async () => {
      try {
        const response = await fetch('./maps/world.geojson');
        if (!response.ok) throw new Error('Bundled country boundaries are missing');
        const world = await response.json();
        if (!instance.current) return;
        map.addSource('countries', { type: 'geojson', data: world, promoteId: 'code' });
        map.addLayer({ id: 'countries', type: 'fill', source: 'countries', paint: { 'fill-color': missingColor, 'fill-opacity': .96 } });
        map.addLayer({ id: 'borders', type: 'line', source: 'countries', paint: { 'line-color': '#f8faf5', 'line-width': .8 } });
        map.addLayer({ id: 'selection', type: 'line', source: 'countries', filter: ['==', ['get', 'code'], 'SE'], paint: { 'line-color': '#153e34', 'line-width': 2.5 } });
        resetWorld();
        map.on('click', 'countries', e => { const p = e.features?.[0].properties; if (p) select.current(p.code, p.name); });
        map.on('mousemove', 'countries', e => {
          map.getCanvas().style.cursor = 'pointer';
          const p = e.features?.[0].properties;
          if (p) setHover({ x: e.point.x, y: e.point.y, code: p.code, name: p.name });
        });
        map.on('mouseleave', 'countries', () => { map.getCanvas().style.cursor = ''; setHover(null); });
        setLoaded(true);
      } catch (e) { setError(String(e)); }
    });
    const observer = new ResizeObserver(() => map.resize()); observer.observe(element.current);
    return () => { observer.disconnect(); instance.current = null; map.remove(); };
  }, []);
  useEffect(() => {
    if (!loaded || !instance.current) return;
    const expression: unknown[] = ['match', ['get', 'code']];
    Object.entries(paint).forEach(([code, p]) => expression.push(code, p.color));
    expression.push(missingColor);
    instance.current.setPaintProperty('countries', 'fill-color', expression.length > 3 ? expression as ExpressionSpecification : missingColor);
    instance.current.setFilter('selection', ['==', ['get', 'code'], selected]);
  }, [paint, selected, loaded]);
  return <div className="map-shell" data-map-ready={loaded}>
    <div ref={element} className="map-canvas" data-testid="world-map" />
    {error && <div className="map-error">{error}</div>}
    {hover && <div className="map-tooltip" style={{ left: Math.min(hover.x + 14, (element.current?.clientWidth ?? 900) - 230), top: Math.max(12, hover.y - 60) }}><strong>{names[hover.code] ?? hover.name}</strong><span>{values.current[hover.code]?.label ?? 'No data in this release'}</span></div>}
    <div className="map-controls"><button aria-label="Zoom in" onClick={() => instance.current?.zoomIn()}><Plus size={17} /></button><button aria-label="Zoom out" onClick={() => instance.current?.zoomOut()}><Minus size={17} /></button><button aria-label="Reset world view" onClick={resetWorld}><Maximize2 size={15} /></button></div>
    <div className="map-scale-note">WORLD OVERVIEW<span>Scroll to zoom · drag to explore</span></div>
  </div>;
}
