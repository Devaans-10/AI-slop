import PaperCrumple from './components/PaperCrumple'

export default function App() {
  return (
    <div style={{ padding: '2rem 1rem' }}>
      <h1>AI Slop Detector</h1>
      <p>Drag and crumple an article preview before running the detector.</p>
      <PaperCrumple
        src="https://picsum.photos/600/800"
        alt="Article preview"
        width={320}
        height={400}
        sceneHeight={560}
        releaseBehavior="restore"
        crumpleAmount={0.85}
        crumpleDuration={0.55}
        releaseDuration={0.4}
        foldCount={6}
        foldSharpness={0.6}
        wrinkleDepth={0.65}
        creaseStrength={0.18}
        paperColor="#f4f0e8"
        paperTexture={0.08}
        draggable
        returnToOrigin
      />
    </div>
  )
}
