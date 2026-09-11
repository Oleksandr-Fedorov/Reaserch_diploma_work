import { createBrowserRouter, RouterProvider } from 'react-router'
import MainPage from './components/mainPage.tsx'

const router = createBrowserRouter([
	{
		path: '/',
		element: <MainPage />,
	},
	{
		path: '/catalog',
		element: <div>Каталог</div>,
	},
	{
		path: '/admin',
		element: <div>Админка</div>,
	},
])

function App() {
	return <RouterProvider router={router} />
}

export default App