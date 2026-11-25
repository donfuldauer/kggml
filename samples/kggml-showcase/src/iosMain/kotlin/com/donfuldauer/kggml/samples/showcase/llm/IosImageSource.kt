package com.donfuldauer.kggml.samples.showcase.llm

import com.donfuldauer.kggml.samples.showcase.toByteArray
import kotlinx.coroutines.CompletableDeferred
import platform.UIKit.UIApplication
import platform.UIKit.UIImage
import platform.UIKit.UIImageJPEGRepresentation
import platform.UIKit.UIImagePickerController
import platform.UIKit.UIImagePickerControllerDelegateProtocol
import platform.UIKit.UIImagePickerControllerOriginalImage
import platform.UIKit.UIImagePickerControllerSourceType
import platform.UIKit.UINavigationControllerDelegateProtocol
import platform.UIKit.UIViewController
import platform.darwin.NSObject

/**
 * Photo library and camera through `UIImagePickerController`. The picked `UIImage` is encoded to JPEG in memory
 * (`UIImageJPEGRepresentation`) and handed over as bytes; nothing is written to disk. The camera needs
 * `NSCameraUsageDescription` in the host app's `Info.plist`.
 */
class IosImageSource : ImageSource {
    /** UIKit holds picker delegates weakly: keep the active one alive until it finishes. */
    private var activeDelegate: PickerDelegate? = null

    override val canPickFromGallery: Boolean
        get() = UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.UIImagePickerControllerSourceTypePhotoLibrary)

    override val canCapture: Boolean
        get() = UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.UIImagePickerControllerSourceTypeCamera)

    override suspend fun pickFromGallery(): ByteArray? =
        present(UIImagePickerControllerSourceType.UIImagePickerControllerSourceTypePhotoLibrary)

    override suspend fun capture(): ByteArray? = present(UIImagePickerControllerSourceType.UIImagePickerControllerSourceTypeCamera)

    private suspend fun present(sourceType: UIImagePickerControllerSourceType): ByteArray? {
        val root = topViewController() ?: error("No view controller to present the image picker from")
        val result = CompletableDeferred<ByteArray?>()
        val delegate = PickerDelegate(result)
        activeDelegate = delegate
        val picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = delegate
        root.presentViewController(picker, animated = true, completion = null)
        return try {
            result.await()
        } finally {
            if (activeDelegate === delegate) activeDelegate = null
        }
    }

    private fun topViewController(): UIViewController? {
        var controller = UIApplication.sharedApplication.keyWindow?.rootViewController
        while (controller?.presentedViewController != null) controller = controller.presentedViewController
        return controller
    }

    private class PickerDelegate(
        private val result: CompletableDeferred<ByteArray?>,
    ) : NSObject(), UIImagePickerControllerDelegateProtocol, UINavigationControllerDelegateProtocol {
        override fun imagePickerController(picker: UIImagePickerController, didFinishPickingMediaWithInfo: Map<Any?, *>) {
            val image = didFinishPickingMediaWithInfo[UIImagePickerControllerOriginalImage] as? UIImage
            val bytes = image?.let { UIImageJPEGRepresentation(it, 0.9)?.toByteArray() }
            picker.dismissViewControllerAnimated(true, completion = null)
            result.complete(bytes)
        }

        override fun imagePickerControllerDidCancel(picker: UIImagePickerController) {
            picker.dismissViewControllerAnimated(true, completion = null)
            result.complete(null)
        }
    }
}
